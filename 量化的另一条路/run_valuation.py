# -*- coding: utf-8 -*-
"""估值约束改进版 — A-share 市场 PB 分位数作为估值校准层"""
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

# ============================================================================
#  1. 拉取 A 股市场 PB 数据 + 计算滚动分位数
# ============================================================================
print("[1/4] 拉取市场 PB 数据...")
import akshare as ak
pb_raw = ak.stock_a_all_pb()
pb_df = pb_raw[["date", "middlePB", "close"]].copy()
pb_df["date"] = pd.to_datetime(pb_df["date"])
pb_df = pb_df.set_index("date").sort_index()
pb_df = pb_df[pb_df["middlePB"] > 0]

# 滚动 10 年分位数（2500 个交易日）
ROLLING_WINDOW = 2500
pb_df["pb_pct_10yr"] = pb_df["middlePB"].rolling(ROLLING_WINDOW, min_periods=500).apply(
    lambda x: (x < x.iloc[-1]).mean(), raw=False
)
print(f"  PB 数据: {len(pb_df)} 行, {pb_df.index[0].date()} ~ {pb_df.index[-1].date()}")
print(f"  最近 PB: {pb_df['middlePB'].iloc[-1]:.2f}, 10yr 分位数: {pb_df['pb_pct_10yr'].iloc[-1]:.0%}")

# ============================================================================
#  2. 加载 ETF 价格矩阵（复用缓存）
# ============================================================================
print("\n[2/4] 加载 ETF 缓存...")
ETF_UNIVERSE = {
    "510300": "沪深300", "510050": "上证50", "510500": "中证500",
    "512100": "中证1000", "159949": "创业板50", "588000": "科创50",
    "159928": "消费", "512010": "医药", "512880": "证券", "512800": "银行",
    "512660": "军工", "515790": "光伏", "159995": "芯片", "516160": "新能源",
    "513100": "纳指", "513050": "中概互联", "513180": "恒生科技",
    "518880": "黄金", "511010": "国债", "511260": "十年国债", "511380": "可转债",
}
raw_data = {}
for code in ETF_UNIVERSE:
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

# ============================================================================
#  3. 策略类（复用原始 + 新估值版）
# ============================================================================
print("\n[3/4] 回测...")

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
        return weekly[::2] if self.rebalance_freq == "2W" else weekly

    def _trend_filter(self, px_slice, bench):
        if bench not in px_slice.columns: return self.bear_equity_cap
        bp = px_slice[bench].dropna()
        if len(bp) < self.trend_ma: return self.bear_equity_cap
        ma = bp.rolling(self.trend_ma).mean().iloc[-1]
        return 1.0 if bp.iloc[-1] > ma else self.bear_equity_cap

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
    """
    估值约束版：在市场 PB 分位数过高时降低权益仓位。

    估值系数 = f(PB 分位数):
      - PB < 20% percentile: 1.0  (便宜，满仓)
      - PB 20-50%:            0.9  (正常偏贵)
      - PB 50-80%:            0.7  (偏贵)
      - PB > 80%:             0.50 (很贵，减半)
    """
    def __init__(self, pb_series=None, **kw):
        super().__init__(**kw)
        self.pb_series = pb_series

    def _valuation_scalar(self, dt):
        """根据调仓日的 PB 分位数计算估值系数。"""
        if self.pb_series is None or len(self.pb_series) == 0:
            return 1.0
        # 找到 dt 之前最近的 PB 分位数
        available = self.pb_series[self.pb_series.index <= dt]
        if len(available) == 0:
            return 1.0
        pct = available.iloc[-1]
        if pd.isna(pct):
            return 1.0
        if pct < 0.20:
            return 1.0    # 便宜
        elif pct < 0.50:
            return 0.90   # 正常偏高
        elif pct < 0.80:
            return 0.70   # 偏贵
        else:
            return 0.50   # 很贵

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
            val_scalar = self._valuation_scalar(dt)
            eq_w = eq_cap * vs * val_scalar
            ms = self._mom_score(ps)
            sel = self._corr_filter(ms, rs)
            if len(sel) == 0: sel, wts = [def_code], {def_code: 1.0}; eq_w = 0.0
            else:
                pw = eq_w / len(sel); wts = {c: pw for c in sel}
                if eq_w < 1.0: wts[def_code] = 1.0 - eq_w
            sigs.append(dict(date=dt, equity_cap=eq_cap, vol_scalar=vs,
                             val_scalar=val_scalar, equity_weight=eq_w,
                             selected=sel, weights=wts))
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
        cagr = (1 + tr_ret) ** (1 / yrs) - 1
        av = rets.std() * np.sqrt(252)
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


# ============================================================================
#  4. 回测：原始 vs 估值约束
# ============================================================================
DEFAULT = dict(mom_windows=(60, 120), top_n=5, trend_ma=200, bear_equity_cap=0.30,
               target_vol=0.15, vol_window=20, corr_threshold=0.80, corr_window=60, rebalance_freq="2W")

print("  原始策略...")
s0 = ETFRotationStrategy(**DEFAULT)
sig0 = s0.generate_signals(px)
r0 = BacktestEngine(100_000).run(px, sig0)
m0, n0, t0 = r0["metrics"], r0["nav"], r0["trades"]

print("  估值约束策略...")
s_val = ValuationStrategy(pb_series=pb_df["pb_pct_10yr"], **DEFAULT)
sig_val = s_val.generate_signals(px)
r_val = BacktestEngine(100_000).run(px, sig_val)
m_val, n_val, t_val = r_val["metrics"], r_val["nav"], r_val["trades"]

# ============================================================================
#  5. 输出
# ============================================================================
print("\n" + "=" * 72)
print("  估值约束回测结果")
print("=" * 72)

compare = OrderedDict()
compare["原始 (无估值)"] = (m0, t0)
compare["估值约束 (PB分位数)"] = (m_val, t_val)

rows = [("年化收益", "cagr"), ("年化波动", "annual_vol"), ("夏普比率", "sharpe"),
        ("最大回撤", "max_drawdown"), ("Calmar", "calmar"), ("月胜率", "win_rate_monthly"),
        ("超额年化", "excess_cagr"), ("信息比率", "ir")]

hdr = f"{'指标':<14s}  {'原始 (无估值)':>20s}  {'估值约束 (PB分位数)':>22s}"
print(hdr); print("-" * len(hdr))
for lb, ky in rows:
    ov = m0.get(ky); nv = m_val.get(ky)
    def fmt(v, k=ky):
        if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
        if k not in ("sharpe", "calmar", "ir", "years", "profit_factor"):
            return f"{v*100:7.2f}%"
        return f"{v:7.2f}"
    print(f"{lb:<14s}  {fmt(ov):>20s}  {fmt(nv):>22s}")
print(f"{'交易次数':<14s}  {len(t0):>20d}  {len(t_val):>22d}")
print(f"{'交易成本':<14s}  {t0['cost'].sum():>18.0f} RMB  {t_val['cost'].sum():>20.0f} RMB")
print("=" * 72)

ex0 = m0.get("excess_cagr", 0) or 0; ex_val = m_val.get("excess_cagr", 0) or 0
print(f"\n最大回撤: {m0['max_drawdown']*100:.1f}% → {m_val['max_drawdown']*100:.1f}%")
print(f"超额年化: {ex0*100:+.2f}% → {ex_val*100:+.2f}%  (变化 {ex_val*100-ex0*100:+.2f}pp)")

# ============================================================================
#  6. 估值信号分布
# ============================================================================
print("\n" + "=" * 72)
print("  估值层信号分布")
print("=" * 72)
if not sig_val.empty:
    pct_bins = pd.cut(sig_val["val_scalar"], bins=[0, 0.55, 0.75, 0.95, 1.05],
                      labels=["0.50 (贵)", "0.70 (偏贵)", "0.90 (正常)", "1.00 (便宜)"])
    print(pct_bins.value_counts().to_string())
    print(f"\n  平均估值系数: {sig_val['val_scalar'].mean():.2f}")
    print(f"  平均权益仓位: {sig_val['equity_weight'].mean()*100:.1f}%  "
          f"(原始: {sig0['equity_weight'].mean()*100:.1f}%)")

# 年份仓位对比
sig0_yr = sig0.copy(); sig0_yr["year"] = sig0_yr.index.year
sig_val_yr = sig_val.copy(); sig_val_yr["year"] = sig_val_yr.index.year
print(f"\n{'Year':<6s}  {'原始仓位':>10s}  {'估值仓位':>10s}  {'差值':>10s}  {'PB分位':>10s}")
print("-" * 52)
for yr in sorted(set(sig0_yr["year"].unique()) | set(sig_val_yr["year"].unique())):
    ow = sig0_yr[sig0_yr["year"] == yr]["equity_weight"].mean()
    nw = sig_val_yr[sig_val_yr["year"] == yr]["equity_weight"].mean()
    # Average PB percentile for that year
    yr_dates = sig_val_yr[sig_val_yr["year"] == yr].index
    pb_pcts = []
    for d in yr_dates:
        avail = pb_df["pb_pct_10yr"][pb_df.index <= d]
        if len(avail) > 0:
            pb_pcts.append(avail.iloc[-1])
    avg_pb_pct = np.mean(pb_pcts) if pb_pcts else 0
    print(f"{yr:<6d}  {ow*100:9.0f}%  {nw*100:9.0f}%  {(nw-ow)*100:+9.0f}pp  {avg_pb_pct*100:9.0f}%")

# ============================================================================
#  7. 图表
# ============================================================================
# Add auxiliary series
if "510300" in px.columns:
    n0["bench_nav"] = px["510300"].reindex(n0.index).ffill()
    n_val["bench_nav"] = n0["bench_nav"]
n0["equity_weight"] = sig0["equity_weight"].reindex(n0.index, method="ffill").fillna(1.0)
n_val["equity_weight"] = sig_val["equity_weight"].reindex(n_val.index, method="ffill").fillna(1.0)

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1, 1, 1.5]})

# NAV
ax = axes[0]
ax.plot(n0.index, n0["nav"] / n0["nav"].iloc[0], color="#9ca3af", lw=1.0, alpha=0.6, label="Original")
ax.plot(n_val.index, n_val["nav"] / n_val["nav"].iloc[0], color="#7c3aed", lw=1.5, label="Valuation")
if "bench_nav" in n0.columns:
    ax.plot(n0.index, n0["bench_nav"] / n0["bench_nav"].iloc[0], color="#d4d4d8", lw=0.8, alpha=0.5, label="CSI 300")
ax.set_ylabel("NAV"); ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
ax.set_title(f"Original (CAGR {m0['cagr']*100:.1f}%, Sharpe {m0['sharpe']:.2f})  "
             f"vs  Valuation (CAGR {m_val['cagr']*100:.1f}%, Sharpe {m_val['sharpe']:.2f})")

# DD
ax2 = axes[1]
dd0 = (n0["nav"] - n0["nav"].cummax()) / n0["nav"].cummax()
dd_val = (n_val["nav"] - n_val["nav"].cummax()) / n_val["nav"].cummax()
ax2.fill_between(n0.index, 0, dd0, color="#9ca3af", alpha=0.2, label="Original")
ax2.fill_between(n_val.index, 0, dd_val, color="#7c3aed", alpha=0.3, label="Valuation")
ax2.set_ylabel("Drawdown"); ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax2.legend(loc="lower left"); ax2.grid(True, alpha=0.3)

# Equity weight
ax3 = axes[2]
ax3.plot(n0.index, n0["equity_weight"], color="#9ca3af", lw=0.8, alpha=0.7, label="Original")
ax3.plot(n_val.index, n_val["equity_weight"], color="#7c3aed", lw=1.0, label="Valuation")
ax3.set_ylabel("Equity Weight"); ax3.set_ylim(-0.05, 1.1)
ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax3.legend(loc="upper left"); ax3.grid(True, alpha=0.3)

# PB percentile
ax4 = axes[3]
pb_overlap = pb_df["pb_pct_10yr"].reindex(n_val.index).ffill()
ax4.fill_between(n_val.index, 0, pb_overlap, color="#f59e0b", alpha=0.4)
ax4.axhline(y=0.20, color="#22c55e", lw=1, ls="--", alpha=0.5, label="20% (cheap)")
ax4.axhline(y=0.80, color="#ef4444", lw=1, ls="--", alpha=0.5, label="80% (expensive)")
ax4.set_ylabel("A-share PB %ile"); ax4.set_ylim(-0.05, 1.1)
ax4.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax4.legend(loc="upper left"); ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("valuation_comparison.png", dpi=150, bbox_inches="tight")
print("\n[图表] valuation_comparison.png")
print("\nDone.")
