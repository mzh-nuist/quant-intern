# -*- coding: utf-8 -*-
"""改进版策略回测 — 独立脚本"""
import os, warnings
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

# ============================================================================
#  ETF 池 & 数据加载
# ============================================================================
ETF_UNIVERSE = {
    "510300": {"name": "沪深300ETF"}, "510050": {"name": "上证50ETF"},
    "510500": {"name": "中证500ETF"}, "512100": {"name": "中证1000ETF"},
    "159949": {"name": "创业板50ETF"}, "588000": {"name": "科创50ETF"},
    "159928": {"name": "消费ETF"}, "512010": {"name": "医药ETF"},
    "512880": {"name": "证券ETF"}, "512800": {"name": "银行ETF"},
    "512660": {"name": "军工ETF"}, "515790": {"name": "光伏ETF"},
    "159995": {"name": "芯片ETF"}, "516160": {"name": "新能源ETF"},
    "513100": {"name": "纳指ETF"}, "513050": {"name": "中概互联ETF"},
    "513180": {"name": "恒生科技ETF"}, "518880": {"name": "黄金ETF"},
    "511010": {"name": "国债ETF"}, "511260": {"name": "十年国债ETF"},
    "511380": {"name": "可转债ETF"},
}

print("加载缓存...")
raw_data = {}
for code in ETF_UNIVERSE:
    f = os.path.join("etf_cache", f"{code}.csv")
    if os.path.exists(f):
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        raw_data[code] = df

closes = {c: d["close"] for c, d in raw_data.items()}
amounts = {c: d.get("amount", pd.Series(0, index=d.index)) for c, d in raw_data.items()}
price_matrix = pd.DataFrame(closes).sort_index().dropna(how="all")
am = pd.DataFrame(amounts).reindex(price_matrix.index)
valid = [c for c, a in am.mean().items() if a >= 10_000_000]
price_matrix = price_matrix[valid]
print(f"价格矩阵: {price_matrix.shape[0]}d x {price_matrix.shape[1]} ETFs")

# ============================================================================
#  Strategy classes (minimal)
# ============================================================================
class ETFRotationStrategy:
    def __init__(self, mom_windows=(60, 120), top_n=5, trend_ma=200,
                 bear_equity_cap=0.30, target_vol=0.15, vol_window=20,
                 corr_threshold=0.80, corr_window=60, rebalance_freq="2W"):
        self.mom_windows = mom_windows; self.top_n = top_n
        self.trend_ma = trend_ma; self.bear_equity_cap = bear_equity_cap
        self.target_vol = target_vol; self.vol_window = vol_window
        self.corr_threshold = corr_threshold; self.corr_window = corr_window
        self.rebalance_freq = rebalance_freq

    def generate_signals(self, px, bench_code="510300", def_code="511010"):
        rets = px.pct_change()
        rb_dates = self._get_rb_dates(px.index)
        sigs = []
        for dt in rb_dates:
            try: pos = px.index.get_loc(dt)
            except KeyError: continue
            if pos < max(self.trend_ma, max(self.mom_windows), self.corr_window): continue
            ps = px.iloc[:pos + 1]; rs = rets.iloc[:pos + 1]
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
        if self.rebalance_freq == "W": return weekly
        if self.rebalance_freq == "2W": return weekly[::2]
        return weekly  # fallback

    def _trend_filter(self, px, bench):
        if bench not in px.columns: return self.bear_equity_cap
        bp = px[bench].dropna()
        if len(bp) < self.trend_ma: return self.bear_equity_cap
        ma = bp.rolling(self.trend_ma).mean().iloc[-1]
        return 1.0 if bp.iloc[-1] > ma else self.bear_equity_cap

    def _vol_scalar(self, px, bench):
        if bench not in px.columns: return 1.0
        bp = px[bench].dropna()
        if len(bp) < self.vol_window + 1: return 1.0
        lr = np.log(bp / bp.shift(1)).dropna().iloc[-self.vol_window:]
        av = lr.std() * np.sqrt(252)
        return min(1.0, self.target_vol / av) if av > 0 else 1.0

    def _mom_score(self, px):
        scores = pd.Series(index=px.columns, dtype=float)
        for c in px.columns:
            p = px[c].dropna()
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


class ImprovedStrategy(ETFRotationStrategy):
    def __init__(self, reversal_threshold=0.08, **kw):
        super().__init__(**kw)
        self.reversal_threshold = reversal_threshold

    def _trend_filter(self, px, bench):
        if bench not in px.columns: return self.bear_equity_cap
        bp = px[bench].dropna()
        if len(bp) < max(self.trend_ma, 60): return self.bear_equity_cap
        below_ma = bp.iloc[-1] <= bp.rolling(self.trend_ma).mean().iloc[-1]
        downtrend = (bp.iloc[-1] / bp.iloc[-60] - 1) < 0
        if below_ma and downtrend: return self.bear_equity_cap
        if below_ma or downtrend: return 0.50
        return 1.0

    def _mom_score(self, px):
        scores = pd.Series(index=px.columns, dtype=float)
        for c in px.columns:
            p = px[c].dropna()
            if len(p) < max(self.mom_windows): scores[c] = -np.inf; continue
            if self.reversal_threshold > 0 and len(p) >= 10:
                if p.iloc[-1] / p.iloc[-10] - 1 > self.reversal_threshold:
                    scores[c] = -np.inf; continue
            wins = [p.iloc[-1] / p.iloc[-w] - 1 for w in self.mom_windows if len(p) >= w]
            scores[c] = np.mean(wins) if wins else -np.inf
        return scores.sort_values(ascending=False)


class BacktestEngine:
    def __init__(self, init_cap=100_000, comm=0.00005, slip=0.0005):
        self.init_cap = init_cap; self.comm = comm; self.slip = slip

    def run(self, px, sig):
        px, sig = px.copy(), sig.copy()
        cash = self.init_cap; hld = {}; nvs = []; trs = []
        for date in px.index:
            if date in sig.index:
                t, cash = self._rb(date, sig.loc[date, "weights"], hld, cash, px.loc[date].to_dict())
                if t: trs.extend(t)
            hv = sum(sh * px.loc[date].get(c, 0) for c, sh in hld.items() if np.isfinite(px.loc[date].get(c, np.nan)))
            nvs.append({"date": date, "nav": cash + hv})
        nav = pd.DataFrame(nvs).set_index("date"); nav["returns"] = nav["nav"].pct_change()
        return {"nav": nav, "trades": pd.DataFrame(trs) if trs else pd.DataFrame(),
                "metrics": self._metrics(nav, px)}

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

    def _metrics(self, nav, px):
        rets = nav["returns"].dropna()
        if len(rets) < 20: return {"error": "data too short"}
        yrs = max((nav.index[-1] - nav.index[0]).days / 365.25, 0.5)
        tr = nav["nav"].iloc[-1] / nav["nav"].iloc[0] - 1
        cagr = (1 + tr) ** (1 / yrs) - 1
        av = rets.std() * np.sqrt(252)
        sh = (cagr - 0.02) / av if av > 0 else 0
        dd = ((nav["nav"] - nav["nav"].cummax()) / nav["nav"].cummax()).min()
        cm = cagr / abs(dd) if dd != 0 else 0
        mo = nav["returns"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
        wr = (mo > 0).mean(); aw = mo[mo > 0].mean(); al = mo[mo < 0].mean()
        pf = abs(aw * (mo > 0).sum() / (al * (mo < 0).sum())) if al != 0 and (mo < 0).sum() > 0 else np.inf
        bi = {}
        if "510300" in px.columns:
            b = px["510300"].reindex(nav.index).ffill(); br = b.pct_change().dropna()
            bt = b.iloc[-1] / b.iloc[0] - 1; bc = (1 + bt) ** (1 / yrs) - 1
            te = (rets - br.reindex(rets.index)).dropna().std() * np.sqrt(252)
            bi = dict(bench_cagr=bc, bench_vol=br.std() * np.sqrt(252),
                      bench_max_dd=((b - b.cummax()) / b.cummax()).min(),
                      excess_cagr=cagr - bc, tracking_err=te)
            bi["ir"] = bi["excess_cagr"] / te if te > 0 else 0
        return dict(total_return=tr, cagr=cagr, annual_vol=av, sharpe=sh,
                    max_drawdown=dd, calmar=cm, win_rate_monthly=wr,
                    avg_win_monthly=aw, avg_loss_monthly=al, profit_factor=pf, years=yrs, **bi)


# ============================================================================
#  Run
# ============================================================================
DEFAULT = dict(mom_windows=(60, 120), top_n=5, trend_ma=200, bear_equity_cap=0.30,
               target_vol=0.15, vol_window=20, corr_threshold=0.80, corr_window=60, rebalance_freq="2W")
IMPROVED = dict(mom_windows=(20, 60, 120), top_n=5, trend_ma=150, bear_equity_cap=0.30,
                target_vol=0.15, vol_window=20, corr_threshold=0.80, corr_window=60,
                rebalance_freq="2W", reversal_threshold=0.08)

print("\n[1/3] 原始策略...")
s0 = ETFRotationStrategy(**DEFAULT)
sig0 = s0.generate_signals(price_matrix)
r0 = BacktestEngine(100_000).run(price_matrix, sig0)
m0, n0, t0 = r0["metrics"], r0["nav"], r0["trades"]

print("[2/3] 改进策略...")
s1 = ImprovedStrategy(**IMPROVED)
sig1 = s1.generate_signals(price_matrix)
r1 = BacktestEngine(100_000).run(price_matrix, sig1)
m1, n1, t1 = r1["metrics"], r1["nav"], r1["trades"]

# ========================
#  Head-to-head
# ========================
print("\n" + "=" * 72)
print("  头 对 头 对 比")
print("=" * 72)
rows = [("年化收益", "cagr"), ("年化波动", "annual_vol"), ("夏普比率", "sharpe"),
        ("最大回撤", "max_drawdown"), ("Calmar", "calmar"), ("月胜率", "win_rate_monthly"),
        ("超额年化", "excess_cagr"), ("信息比率", "ir")]
hdr = f"{'指标':<14s}  {'原始 (MA200,60+120)':>24s}  {'改进 (MA150双确认,20+60+120,反转避免)':>30s}"
print(hdr); print("-" * len(hdr))
for lb, ky in rows:
    ov = m0.get(ky); nv = m1.get(ky)
    def fmt(v, is_pct=True):
        if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
        if is_pct and ky not in ("sharpe", "calmar", "ir"):
            return f"{v*100:7.2f}%"
        return f"{v:7.2f}"
    print(f"{lb:<14s}  {fmt(ov):>24s}  {fmt(nv):>30s}")
print(f"{'交易次数':<14s}  {len(t0):>24d}  {len(t1):>30d}")
print(f"{'交易成本':<14s}  {t0['cost'].sum():>22.0f} RMB  {t1['cost'].sum():>28.0f} RMB")
print("=" * 72)
ex0 = m0.get("excess_cagr", 0) or 0; ex1 = m1.get("excess_cagr", 0) or 0
print(f"\n最大回撤: {m0['max_drawdown']*100:.1f}% → {m1['max_drawdown']*100:.1f}%")
print(f"超额年化: {ex0*100:+.2f}% → {ex1*100:+.2f}%  (变化 {ex1*100-ex0*100:+.2f}pp)")

# ========================
#  Decomposition
# ========================
print("\n" + "=" * 72)
print("  分解贡献")
print("=" * 72)
variants = OrderedDict()
variants["0_原始"] = ("原始", ETFRotationStrategy, DEFAULT)
p_trend = dict(DEFAULT, trend_ma=150)
variants["1_趋势"] = ("仅趋势过滤", ImprovedStrategy,
                       dict(p_trend, mom_windows=(60, 120), reversal_threshold=0))
variants["2_反转"] = ("仅反转避免", ImprovedStrategy,
                       dict(DEFAULT, trend_ma=200, mom_windows=(60, 120), reversal_threshold=0.08))
variants["3_动量"] = ("仅动量窗口", ETFRotationStrategy,
                       dict(DEFAULT, mom_windows=(20, 60, 120)))
variants["4_全部"] = ("全部合并", ImprovedStrategy, IMPROVED)

print(f"{'变体':<18s}  {'CAGR':>8s}  {'Sharpe':>8s}  {'MaxDD':>8s}  {'超额':>8s}")
print("-" * 58)
rd = {}
for k, (lb, cls, params) in variants.items():
    vk = ["mom_windows", "top_n", "trend_ma", "bear_equity_cap",
          "target_vol", "corr_threshold", "rebalance_freq",
          "reversal_threshold", "vol_window", "corr_window"]
    s = cls(**{kk: vv for kk, vv in params.items() if kk in vk})
    sig = s.generate_signals(price_matrix)
    r = BacktestEngine(100_000).run(price_matrix, sig)
    m = r["metrics"]; rd[k] = m
    c = m.get("cagr", 0) or 0; sh = m.get("sharpe", 0) or 0
    dd = m.get("max_drawdown", 0) or 0; ex = m.get("excess_cagr", 0) or 0
    print(f"{lb:<18s}  {c*100:7.2f}%  {sh:7.2f}  {dd*100:7.1f}%  {ex*100:+7.2f}%")
print("-" * 58)
be = rd["0_原始"].get("excess_cagr", 0) or 0
print(f"\n边际贡献 (vs 原始超额 {be*100:+.2f}%):")
for k, lb in [("1_趋势", "趋势过滤"), ("2_反转", "反转避免"),
              ("3_动量", "动量窗口"), ("4_全部", "全部合并")]:
    ex = rd[k].get("excess_cagr", 0) or 0
    print(f"  {lb}: {(ex - be) * 100:+.2f}pp")

# ========================
#  Signal comparison
# ========================
print("\n" + "=" * 72)
print("  信号分布对比")
print("=" * 72)
bull0 = (sig0["equity_cap"] >= 1.0).sum(); bear0 = len(sig0) - bull0
bull1 = (sig1["equity_cap"] >= 1.0).sum()
weak1 = ((sig1["equity_cap"] > 0.30) & (sig1["equity_cap"] < 1.0)).sum()
bear1 = (sig1["equity_cap"] <= 0.30).sum()
print(f"原始: 牛市 {bull0}/{len(sig0)} ({bull0/len(sig0)*100:.0f}%)  熊市 {bear0}/{len(sig0)} ({bear0/len(sig0)*100:.0f}%)")
print(f"改进: 牛市 {bull1}/{len(sig1)} ({bull1/len(sig1)*100:.0f}%)  "
      f"弱熊 {weak1} ({weak1/len(sig1)*100:.0f}%)  强熊 {bear1} ({bear1/len(sig1)*100:.0f}%)")

sig0_yr = sig0.copy(); sig0_yr["year"] = sig0_yr.index.year
sig1_yr = sig1.copy(); sig1_yr["year"] = sig1_yr.index.year
print(f"\n{'Year':<6s}  {'原始仓位':>10s}  {'改进仓位':>10s}  {'差值':>10s}")
print("-" * 42)
for yr in sorted(set(sig0_yr["year"].unique()) | set(sig1_yr["year"].unique())):
    ow = sig0_yr[sig0_yr["year"] == yr]["equity_weight"].mean()
    nw = sig1_yr[sig1_yr["year"] == yr]["equity_weight"].mean()
    print(f"{yr:<6d}  {ow*100:9.0f}%  {nw*100:9.0f}%  {(nw-ow)*100:+9.0f}pp")

# ========================
#  Chart
# ========================
if "510300" in price_matrix.columns:
    n0["bench_nav"] = price_matrix["510300"].reindex(n0.index).ffill()
    n1["bench_nav"] = n0["bench_nav"]
n0["equity_weight"] = sig0["equity_weight"].reindex(n0.index, method="ffill").fillna(1.0)
n1["equity_weight"] = sig1["equity_weight"].reindex(n1.index, method="ffill").fillna(1.0)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1, 1]})
# NAV
ax = axes[0]
ax.plot(n0.index, n0["nav"] / n0["nav"].iloc[0], color="#9ca3af", lw=1.0, alpha=0.7, label="Original")
ax.plot(n1.index, n1["nav"] / n1["nav"].iloc[0], color="#2563eb", lw=1.5, label="Improved")
if "bench_nav" in n0.columns:
    ax.plot(n0.index, n0["bench_nav"] / n0["bench_nav"].iloc[0], color="#d4d4d8", lw=0.8, alpha=0.5, label="CSI 300")
ax.set_ylabel("NAV"); ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
ax.set_title(f"Original (CAGR {m0['cagr']*100:.1f}%, Sharpe {m0['sharpe']:.2f})  "
             f"→  Improved (CAGR {m1['cagr']*100:.1f}%, Sharpe {m1['sharpe']:.2f})")
# DD
ax2 = axes[1]
dd0 = (n0["nav"] - n0["nav"].cummax()) / n0["nav"].cummax()
dd1 = (n1["nav"] - n1["nav"].cummax()) / n1["nav"].cummax()
ax2.fill_between(n0.index, 0, dd0, color="#9ca3af", alpha=0.25, label="Original")
ax2.fill_between(n1.index, 0, dd1, color="#2563eb", alpha=0.35, label="Improved")
ax2.set_ylabel("Drawdown"); ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax2.legend(loc="lower left"); ax2.grid(True, alpha=0.3)
# Equity weight
ax3 = axes[2]
ax3.plot(n0.index, n0["equity_weight"], color="#9ca3af", lw=0.8, alpha=0.7, label="Original")
ax3.plot(n1.index, n1["equity_weight"], color="#2563eb", lw=1.0, label="Improved")
ax3.set_ylabel("Equity Weight"); ax3.set_ylim(-0.05, 1.1)
ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax3.legend(loc="upper left"); ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("comparison_chart.png", dpi=150, bbox_inches="tight")
print("\n[图表] comparison_chart.png")

print("\nDone — all improvements tested.")
