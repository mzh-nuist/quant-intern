# -*- coding: utf-8 -*-
"""最终版：ETF池优化 + 估值约束，合并回测"""
import os, sys, warnings
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
#  0. 配置
# ============================================================
# 优化后的池子：去除上市晚/数据不足的 ETF
#   - 588000 科创50    (2020-11) → 移除
#   - 513180 恒生科技  (2021-05) → 移除
#   - 515790 光伏      (2020-12) → 移除
#   - 516160 新能源    (2021-02) → 移除
#  保留 17 只，全部在 2020-01-02 或附近有数据
ETF_POOL_V3 = {
    # 宽基
    "510300": "沪深300", "510050": "上证50", "510500": "中证500",
    "512100": "中证1000", "159949": "创业板50",
    # 行业-防御
    "159928": "消费", "512010": "医药",
    # 行业-周期
    "512880": "证券", "512800": "银行",
    # 行业-成长
    "512660": "军工", "159995": "芯片",
    # 跨境
    "513100": "纳指", "513050": "中概互联",
    # 商品
    "518880": "黄金",
    # 债券
    "511010": "国债", "511260": "十年国债", "511380": "可转债",
}

print("优化后 ETF 池: 17 只")
print(f"移除: 科创50(588000), 恒生科技(513180), 光伏(515790), 新能源(516160)")
print(f"理由: 上市晚，前段数据缺失，动量排名失真")

# ============================================================
#  1. 加载 ETF 缓存 + 拉取 PB 数据
# Bypass system proxy (it's unreliable)
import requests as _requests
_old_session_get = _requests.Session.get
def _no_proxy_get(self, url, **kw):
    kw["proxies"] = {}
    kw["timeout"] = kw.get("timeout", 15)
    return _old_session_get(self, url, **kw)
_requests.Session.get = _no_proxy_get
# ============================================================
print("\n[1/4] 加载数据...")
raw_data = {}
for code in ETF_POOL_V3:
    f = os.path.join("etf_cache", f"{code}.csv")
    if os.path.exists(f):
        raw_data[code] = pd.read_csv(f, index_col=0, parse_dates=True)
closes = {c: d["close"] for c, d in raw_data.items()}
amounts = {c: d.get("amount", pd.Series(0, index=d.index)) for c, d in raw_data.items()}
px = pd.DataFrame(closes).sort_index().dropna(how="all")
am = pd.DataFrame(amounts).reindex(px.index)
valid = [c for c, a in am.mean().items() if a >= 10_000_000]
px = px[valid]
print(f"  价格矩阵: {px.shape[0]}d x {px.shape[1]} ETFs, {px.index[0].date()} ~ {px.index[-1].date()}")

# PB 分位数
print("  拉取 A 股 PB 分位数...")
import akshare as ak
pb_raw = ak.stock_a_all_pb()[["date", "middlePB", "close"]]
pb_raw["date"] = pd.to_datetime(pb_raw["date"])
pb_df = pb_raw.set_index("date").sort_index()
pb_df = pb_df[pb_df["middlePB"] > 0]
W = 2500
pb_df["pb_pct"] = pb_df["middlePB"].rolling(W, min_periods=500).apply(
    lambda x: (x < x.iloc[-1]).mean(), raw=False
)
print(f"  最近 PB 分位数: {pb_df['pb_pct'].iloc[-1]:.0%}")

# ============================================================
#  2. 策略类
# ============================================================
class ETFRotationStrategy:
    def __init__(self, mom_windows=(60, 120), top_n=5, trend_ma=200,
                 bear_equity_cap=0.30, target_vol=0.15, vol_window=20,
                 corr_threshold=0.80, corr_window=60, rebalance_freq="2W"):
        self.mom_windows = mom_windows; self.top_n = top_n
        self.trend_ma = trend_ma; self.bear_equity_cap = bear_equity_cap
        self.target_vol = target_vol; self.vol_window = vol_window
        self.corr_threshold = corr_threshold; self.corr_window = corr_window
        self.rebalance_freq = rebalance_freq

    def generate_signals(self, px_mat, bench_code="510300", def_code="511010"):
        rets = px_mat.pct_change()
        rb_dates = self._get_rb_dates(px_mat.index)
        sigs = []
        for dt in rb_dates:
            try: pos = px_mat.index.get_loc(dt)
            except KeyError: continue
            if pos < max(self.trend_ma, max(self.mom_windows), self.corr_window): continue
            ps = px_mat.iloc[:pos + 1]; rs = rets.iloc[:pos + 1]
            eq_cap = self._trend_filter(ps, bench_code)
            vs = self._vol_scalar(ps, bench_code)
            eq_w = eq_cap * vs
            ms = self._mom_score(ps)
            sel = self._corr_filter(ms, rs)
            if len(sel) == 0: sel, wts = [def_code], {def_code: 1.0}; eq_w = 0.0
            else:
                pw = eq_w / len(sel); wts = {c: pw for c in sel}
                if eq_w < 1.0: wts[def_code] = 1.0 - eq_w
            sigs.append(dict(date=dt, equity_cap=eq_cap, vol_scalar=vs,
                             equity_weight=eq_w, selected=sel, weights=wts))
        return pd.DataFrame(sigs).set_index("date")

    def _get_rb_dates(self, dates):
        iso = dates.isocalendar()
        y = iso["year"] if isinstance(iso, pd.DataFrame) else iso.year
        w = iso["week"] if isinstance(iso, pd.DataFrame) else iso.week
        wid = y.values * 100 + w.values
        df = pd.DataFrame({"d": dates, "wid": wid})
        weekly = pd.DatetimeIndex(df.groupby("wid")["d"].last().sort_values())
        return weekly[::2]

    def _trend_filter(self, px_slice, bench):
        if bench not in px_slice.columns: return self.bear_equity_cap
        bp = px_slice[bench].dropna()
        if len(bp) < self.trend_ma: return self.bear_equity_cap
        return 1.0 if bp.iloc[-1] > bp.rolling(self.trend_ma).mean().iloc[-1] else self.bear_equity_cap

    def _vol_scalar(self, px_slice, bench):
        if bench not in px_slice.columns: return 1.0
        bp = px_slice[bench].dropna()
        if len(bp) < self.vol_window + 1: return 1.0
        lr = np.log(bp / bp.shift(1)).dropna().iloc[-self.vol_window:]
        av = lr.std() * np.sqrt(252)
        return min(1.0, self.target_vol / av) if av > 0 else 1.0

    def _mom_score(self, px_slice):
        scores = pd.Series(index=px_slice.columns, dtype=float)
        for c in px_slice.columns:
            p = px_slice[c].dropna()
            if len(p) < max(self.mom_windows): scores[c] = -np.inf; continue
            wins = [p.iloc[-1] / p.iloc[-w] - 1 for w in self.mom_windows if len(p) >= w]
            scores[c] = np.mean(wins) if wins else -np.inf
        return scores.sort_values(ascending=False)

    def _corr_filter(self, ms, rs):
        cr = rs.iloc[-self.corr_window:]; sel = []
        for c in ms.index:
            if ms[c] == -np.inf or c not in cr.columns: continue
            skip = False
            for s in sel:
                pair = cr[[c, s]].dropna()
                if len(pair) >= 20 and abs(pair.corr().iloc[0, 1]) > self.corr_threshold:
                    skip = True; break
            if not skip: sel.append(c)
            if len(sel) >= self.top_n: break
        return sel


class ValuationStrategy(ETFRotationStrategy):
    """叠加 PB 分位数估值约束。"""
    def __init__(self, pb_pct_series=None, **kw):
        super().__init__(**kw)
        self.pb_pct = pb_pct_series

    def _val_scalar(self, dt):
        if self.pb_pct is None or len(self.pb_pct) == 0:
            return 1.0
        avail = self.pb_pct[self.pb_pct.index <= dt]
        if len(avail) == 0: return 1.0
        p = avail.iloc[-1]
        if pd.isna(p): return 1.0
        if p < 0.20: return 1.0
        if p < 0.50: return 0.90
        if p < 0.80: return 0.70
        return 0.50

    def generate_signals(self, px_mat, bench_code="510300", def_code="511010"):
        rets = px_mat.pct_change()
        rb_dates = self._get_rb_dates(px_mat.index)
        sigs = []
        for dt in rb_dates:
            try: pos = px_mat.index.get_loc(dt)
            except KeyError: continue
            if pos < max(self.trend_ma, max(self.mom_windows), self.corr_window): continue
            ps = px_mat.iloc[:pos + 1]; rs = rets.iloc[:pos + 1]
            eq_w = self._trend_filter(ps, bench_code) * self._vol_scalar(ps, bench_code) * self._val_scalar(dt)
            sel = self._corr_filter(self._mom_score(ps), rs)
            if len(sel) == 0: sel, wts = [def_code], {def_code: 1.0}; eq_w = 0.0
            else:
                pw = eq_w / len(sel); wts = {c: pw for c in sel}
                if eq_w < 1.0: wts[def_code] = 1.0 - eq_w
            sigs.append(dict(date=dt, equity_weight=eq_w, selected=sel, weights=wts))
        return pd.DataFrame(sigs).set_index("date")


class BacktestEngine:
    def __init__(self, init_cap=100_000, comm=0.00005, slip=0.0005):
        self.init_cap = init_cap; self.comm = comm; self.slip = slip

    def run(self, px_mat, sig):
        px_mat, sig = px_mat.copy(), sig.copy()
        cash = self.init_cap; hld = {}; nvs = []; trs = []
        for date in px_mat.index:
            if date in sig.index:
                t, cash = self._rb(date, sig.loc[date, "weights"], hld, cash, px_mat.loc[date].to_dict())
                if t: trs.extend(t)
            hv = sum(sh * px_mat.loc[date].get(c, 0) for c, sh in hld.items()
                     if np.isfinite(px_mat.loc[date].get(c, np.nan)))
            nvs.append({"date": date, "nav": cash + hv})
        nav = pd.DataFrame(nvs).set_index("date"); nav["returns"] = nav["nav"].pct_change()
        return {"nav": nav, "trades": pd.DataFrame(trs) if trs else pd.DataFrame(),
                "metrics": self._metrics(nav, px_mat)}

    def _rb(self, date, tw, hld, cash, prices):
        tr = []
        if not tw: return tr, cash
        cn = cash + sum(sh * prices.get(c, 0) for c, sh in hld.items())
        th = {c: cn * w for c, w in tw.items() if w > 0}
        for c in list(hld.keys()):
            if c not in th:
                sh = hld.pop(c); p = prices.get(c, np.nan)
                if np.isfinite(p) and p > 0 and sh > 0:
                    ep = p * (1 - self.slip); pro = sh * ep; co = sh * p * self.comm
                    cash += pro - co
                    tr.append(dict(date=date, code=c, action="sell", shares=sh, price=ep, proceeds=pro, cost=co))
        for c, tv in th.items():
            p = prices.get(c, np.nan)
            if not np.isfinite(p) or p <= 0: continue
            cv = hld.get(c, 0) * p; diff = tv - cv
            if abs(diff) < cn * 0.005: continue
            if diff > 0:
                ep = p * (1 + self.slip); bv = min(diff, cash); nb = int(bv / ep)
                if nb > 0:
                    co = nb * ep; com = co * self.comm; cash -= co + com
                    hld[c] = hld.get(c, 0) + nb
                    tr.append(dict(date=date, code=c, action="buy", shares=nb, price=ep, proceeds=-co, cost=com))
            else:
                ep = p * (1 - self.slip); ns = min(int(abs(diff) / p), hld.get(c, 0))
                if ns > 0:
                    pro = ns * ep; com = pro * self.comm; cash += pro - com
                    hld[c] -= ns
                    if hld[c] == 0: del hld[c]
                    tr.append(dict(date=date, code=c, action="sell", shares=ns, price=ep, proceeds=pro, cost=com))
        return tr, cash

    def _metrics(self, nav, px_mat):
        rets = nav["returns"].dropna()
        if len(rets) < 20: return {"error": "data too short"}
        yrs = max((nav.index[-1] - nav.index[0]).days / 365.25, 0.5)
        tr_ret = nav["nav"].iloc[-1] / nav["nav"].iloc[0] - 1
        cagr = (1 + tr_ret) ** (1 / yrs) - 1; av = rets.std() * np.sqrt(252)
        sh = (cagr - 0.02) / av if av > 0 else 0
        dd = ((nav["nav"] - nav["nav"].cummax()) / nav["nav"].cummax()).min()
        cm = cagr / abs(dd) if dd != 0 else 0
        mo = nav["returns"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
        wr = (mo > 0).mean(); aw = mo[mo > 0].mean(); al = mo[mo < 0].mean()
        pf = abs(aw * (mo > 0).sum() / (al * (mo < 0).sum())) if al != 0 and (mo < 0).sum() > 0 else np.inf
        bi = {}
        if "510300" in px_mat.columns:
            b = px_mat["510300"].reindex(nav.index).ffill(); br = b.pct_change().dropna()
            bt = b.iloc[-1] / b.iloc[0] - 1; bc = (1 + bt) ** (1 / yrs) - 1
            te = (rets - br.reindex(rets.index)).dropna().std() * np.sqrt(252)
            bi = dict(bench_cagr=bc, bench_vol=br.std() * np.sqrt(252),
                      bench_max_dd=((b - b.cummax()) / b.cummax()).min(),
                      excess_cagr=cagr - bc, tracking_err=te)
            bi["ir"] = bi["excess_cagr"] / te if te > 0 else 0
        return dict(total_return=tr_ret, cagr=cagr, annual_vol=av, sharpe=sh,
                    max_drawdown=dd, calmar=cm, win_rate_monthly=wr,
                    avg_win_monthly=aw, avg_loss_monthly=al, profit_factor=pf, years=yrs, **bi)


# ============================================================
#  3. 回测：三个版本
# ============================================================
DEFAULT = dict(mom_windows=(60, 120), top_n=5, trend_ma=200, bear_equity_cap=0.30,
               target_vol=0.15, vol_window=20, corr_threshold=0.80, corr_window=60, rebalance_freq="2W")

print("\n[2/4] 回测三版本...")

# A. 原始池 (21只) + 原始策略
print("  A. 原始池 + 原始策略")
px_old = pd.read_csv("run_improved.py", nrows=0)  # dummy, load from cache
# Actually reload px for old pool:
old_pool_file = "run_improved.py"  # not used, we reconstruct
# For old pool: reload all 21
raw_old = {}
for code in ["510300","510050","510500","512100","159949","588000",
             "159928","512010","512880","512800","512660","515790",
             "159995","516160","513100","513050","513180","518880",
             "511010","511260","511380"]:
    f = os.path.join("etf_cache", f"{code}.csv")
    if os.path.exists(f):
        raw_old[code] = pd.read_csv(f, index_col=0, parse_dates=True)
c_old = {c: d["close"] for c, d in raw_old.items()}
px_old = pd.DataFrame(c_old).sort_index().dropna(how="all")
valid_old = [c for c, a in pd.DataFrame({c: d.get("amount",pd.Series(0,index=d.index)) for c,d in raw_old.items()}).reindex(px_old.index).mean().items() if a >= 10_000_000]
px_old = px_old[valid_old]

sA = ETFRotationStrategy(**DEFAULT)
sigA = sA.generate_signals(px_old)
rA = BacktestEngine(100_000).run(px_old, sigA)
mA, nA, tA = rA["metrics"], rA["nav"], rA["trades"]

# B. 优化池 (17只) + 原始策略
print("  B. 优化池 + 原始策略")
sB = ETFRotationStrategy(**DEFAULT)
sigB = sB.generate_signals(px)
rB = BacktestEngine(100_000).run(px, sigB)
mB, nB, tB = rB["metrics"], rB["nav"], rB["trades"]

# C. 优化池 (17只) + 估值约束
print("  C. 优化池 + 估值约束")
sC = ValuationStrategy(pb_pct_series=pb_df["pb_pct"], **DEFAULT)
sigC = sC.generate_signals(px)
rC = BacktestEngine(100_000).run(px, sigC)
mC, nC, tC = rC["metrics"], rC["nav"], rC["trades"]

# ============================================================
#  4. 三路对比输出
# ============================================================
print("\n" + "=" * 90)
print("  最终对比：原始 vs 池子优化 vs 池子优化+估值约束")
print("=" * 90)

compare = OrderedDict([
    ("A. 原始 (21只)", (mA, tA)),
    ("B. 优化池 (17只)", (mB, tB)),
    ("C. 池优化+估值", (mC, tC)),
])

rows = [("年化收益", "cagr"), ("年化波动", "annual_vol"), ("夏普比率", "sharpe"),
        ("最大回撤", "max_drawdown"), ("Calmar", "calmar"),
        ("月胜率", "win_rate_monthly"), ("超额年化", "excess_cagr"), ("信息比率", "ir")]

hdr = f"{'指标':<14s}"
for name in compare: hdr += f"  {name:<26s}"
print(hdr); print("-" * len(hdr))
for lb, ky in rows:
    line = f"{lb:<14s}"
    for (m, _) in compare.values():
        v = m.get(ky)
        if v is None or (isinstance(v, float) and np.isnan(v)): line += f"  {'N/A':>26s}"
        elif ky in ("sharpe", "calmar", "ir"): line += f"  {v:>26.2f}"
        else: line += f"  {v*100:>25.2f}%"
    print(line)
print(f"{'交易次数':<14s}  {len(tA):>26d}  {len(tB):>26d}  {len(tC):>26d}")
print(f"{'交易成本':<14s}  {tA['cost'].sum():>24.0f} RMB  {tB['cost'].sum():>24.0f} RMB  {tC['cost'].sum():>24.0f} RMB")
print("=" * 90)

# 改进幅度
exA = mA.get("excess_cagr", 0) or 0
exB = mB.get("excess_cagr", 0) or 0
exC = mC.get("excess_cagr", 0) or 0
print(f"\n超额年化: {exA*100:+.2f}% → {exB*100:+.2f}% → {exC*100:+.2f}%")
print(f"池优化边际: {(exB-exA)*100:+.2f}pp")
print(f"估值约束边际: {(exC-exB)*100:+.2f}pp")
print(f"总改进: {(exC-exA)*100:+.2f}pp")

ddA = mA["max_drawdown"]; ddB = mB["max_drawdown"]; ddC = mC["max_drawdown"]
print(f"最大回撤: {ddA*100:.1f}% → {ddB*100:.1f}% → {ddC*100:.1f}%")
shA = mA["sharpe"]; shB = mB["sharpe"]; shC = mC["sharpe"]
print(f"夏普比率: {shA:.2f} → {shB:.2f} → {shC:.2f}")

# ============================================================
#  5. 图表
# ============================================================
print("\n[3/4] 画图...")
# Add bench + equity weight
for n in [nA, nB, nC]:
    if "510300" in px_old.columns:
        n["bench_nav"] = px_old["510300"].reindex(n.index).ffill()
nA["ew"] = sigA["equity_weight"].reindex(nA.index, method="ffill").fillna(1.0)
nB["ew"] = sigB["equity_weight"].reindex(nB.index, method="ffill").fillna(1.0)
nC["ew"] = sigC["equity_weight"].reindex(nC.index, method="ffill").fillna(1.0)

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1, 1, 1.5]})
colors = {"A": "#9ca3af", "B": "#2563eb", "C": "#7c3aed"}
styles = {"A": (1.0, 0.7), "B": (1.2, 1.0), "C": (1.5, 1.0)}

# NAV
ax = axes[0]
for tag, (m, _), n in [("A", (mA, tA), nA), ("B", (mB, tB), nB), ("C", (mC, tC), nC)]:
    ax.plot(n.index, n["nav"]/n["nav"].iloc[0], color=colors[tag], lw=styles[tag][0],
            alpha=styles[tag][1], label=f"{tag}: CAGR {m['cagr']*100:.1f}%, Sh {m['sharpe']:.2f}")
if "bench_nav" in nA.columns:
    ax.plot(nA.index, nA["bench_nav"]/nA["bench_nav"].iloc[0], color="#d4d4d8", lw=0.6, alpha=0.4, label="CSI 300")
ax.set_ylabel("NAV"); ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
ax.set_title("Final: ETF Pool Optimization + Valuation Overlay")

# DD
ax2 = axes[1]
for tag, (m, _), n in [("A", (mA, tA), nA), ("B", (mB, tB), nB), ("C", (mC, tC), nC)]:
    dd = (n["nav"] - n["nav"].cummax()) / n["nav"].cummax()
    ax2.fill_between(n.index, 0, dd, color=colors[tag], alpha=0.2 if tag == "A" else 0.3,
                     label=f"{tag}: MaxDD {m['max_drawdown']*100:.1f}%")
ax2.set_ylabel("Drawdown"); ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax2.legend(loc="lower left", fontsize=8); ax2.grid(True, alpha=0.3)

# Equity weight
ax3 = axes[2]
for tag, n in [("A", nA), ("B", nB), ("C", nC)]:
    ax3.plot(n.index, n["ew"], color=colors[tag], lw=styles[tag][0], alpha=styles[tag][1], label=tag)
ax3.set_ylabel("Equity Wt"); ax3.set_ylim(-0.05, 1.1)
ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax3.legend(loc="upper left", fontsize=8); ax3.grid(True, alpha=0.3)

# PB percentile
ax4 = axes[3]
pb_s = pb_df["pb_pct"].reindex(nA.index).ffill()
ax4.fill_between(nA.index, 0, pb_s, color="#f59e0b", alpha=0.35)
ax4.axhline(y=0.20, color="#22c55e", lw=1, ls="--", alpha=0.4, label="20%")
ax4.axhline(y=0.80, color="#ef4444", lw=1, ls="--", alpha=0.4, label="80%")
ax4.set_ylabel("A-share PB %ile"); ax4.set_ylim(-0.05, 1.1)
ax4.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax4.legend(loc="upper left", fontsize=8); ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("final_comparison.png", dpi=150, bbox_inches="tight")
print("  [图表] final_comparison.png")

# ============================================================
#  6. 信号对比摘要
# ============================================================
print("\n[4/4] 信号分析...")
for tag, sig, label in [("A", sigA, "原始"), ("B", sigB, "优化池"), ("C", sigC, "池优+估值")]:
    eq_mean = sig["equity_weight"].mean()
    bull = (sig["equity_cap"] >= 1.0).sum() if "equity_cap" in sig.columns else "N/A"
    print(f"  {label}: 信号 {len(sig)} 次, 平均仓位 {eq_mean*100:.0f}%, 牛市 {bull}")

# Yearly breakdown for version C
sigC_yr = sigC.copy(); sigC_yr["year"] = sigC_yr.index.year
print(f"\n  版本 C 各年仓位:")
for yr in sorted(sigC_yr["year"].unique()):
    gw = sigC_yr[sigC_yr["year"] == yr]["equity_weight"].mean()
    # Avg PB percentile
    pcts = []
    for d in sigC_yr[sigC_yr["year"] == yr].index:
        a = pb_df["pb_pct"][pb_df.index <= d]
        if len(a) > 0: pcts.append(a.iloc[-1])
    avg_pb = np.mean(pcts) if pcts else 0
    print(f"  {yr}: 仓位 {gw*100:.0f}%, PB分位 {avg_pb*100:.0f}%")

print("\nDone — 最终版回测完成。")
