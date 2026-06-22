# -*- coding: utf-8 -*-
"""扩池回测：17只 vs 20只 (加入红利ETF/豆粕ETF/红利低波ETF)"""
import os, time, warnings
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
#  0. 代理绕过
# ============================================================
for k in list(os.environ.keys()):
    if 'proxy' in k.lower(): del os.environ[k]
import urllib.request
urllib.request.getproxies = lambda: {}

# ============================================================
#  1. 两个ETF池
# ============================================================
POOL_17 = {
    "510300":"沪深300","510050":"上证50","510500":"中证500",
    "512100":"中证1000","159949":"创业板50",
    "159928":"消费","512010":"医药",
    "512880":"证券","512800":"银行",
    "512660":"军工","159995":"芯片",
    "513100":"纳指","513050":"中概互联",
    "518880":"黄金",
    "511010":"国债","511260":"十年国债","511380":"可转债",
}

POOL_20 = {
    **POOL_17,
    "510880":"红利ETF",       # 高股息/价值风格，与成长ETF负相关
    "159985":"豆粕ETF",       # 商品，与股市极低相关
    "512890":"红利低波ETF",   # 价值+低波动，防御属性
}

# ============================================================
#  2. 加载数据
# ============================================================
def load_pool(pool):
    raw = {}
    for code in pool:
        f = os.path.join("etf_cache", f"{code}.csv")
        if os.path.exists(f):
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            raw[code] = df
    closes = {c: d["close"] for c, d in raw.items()}
    amounts = {c: d.get("amount", pd.Series(0, index=d.index)) for c, d in raw.items()}
    px = pd.DataFrame(closes).sort_index().dropna(how="all")
    am = pd.DataFrame(amounts).reindex(px.index)
    valid = [c for c, a in am.mean().items() if a >= 10_000_000]
    px = px[valid]
    return px

print("加载数据...")
px17 = load_pool(POOL_17)
px20 = load_pool(POOL_20)
# 对齐到 2020-01-01 起始（510880 有更早数据，但其他 ETF 从 2020 开始）
px20 = px20[px20.index >= "2020-01-01"]
print(f"  17只池: {px17.shape[0]}d x {px17.shape[1]} ETFs, {px17.index[0].date()}~{px17.index[-1].date()}")
print(f"  20只池: {px20.shape[0]}d x {px20.shape[1]} ETFs, {px20.index[0].date()}~{px20.index[-1].date()}")

# PB分位数
print("  PB分位数...")
import akshare as ak
pb_raw = ak.stock_a_all_pb()[["date","middlePB","close"]]
pb_raw["date"] = pd.to_datetime(pb_raw["date"])
pb_df = pb_raw.set_index("date").sort_index()
pb_df = pb_df[pb_df["middlePB"] > 0]
pb_df["pb_pct"] = pb_df["middlePB"].rolling(2500, min_periods=500).apply(
    lambda x: (x < x.iloc[-1]).mean(), raw=False)

# ============================================================
#  3. 策略类 (估值版)
# ============================================================
class ETFRotationStrategy:
    def __init__(self, mom_windows=(60,120), top_n=5, trend_ma=200,
                 bear_equity_cap=0.30, target_vol=0.15, vol_window=20,
                 corr_threshold=0.80, corr_window=60, rebalance_freq="2W"):
        self.mw=mom_windows;self.top_n=top_n;self.trend_ma=trend_ma
        self.bear_cap=bear_equity_cap;self.tvol=target_vol
        self.vw=vol_window;self.corr_th=corr_threshold
        self.cw=corr_window;self.freq=rebalance_freq

    def generate_signals(self, px, bench="510300", defensive="511010"):
        ret=px.pct_change();rbd=self._rbd(px.index);sigs=[]
        for dt in rbd:
            try:p=px.index.get_loc(dt)
            except KeyError:continue
            if p<max(self.trend_ma,max(self.mw),self.cw):continue
            ps=px.iloc[:p+1];rs=ret.iloc[:p+1]
            ew=self._trend(ps,bench)*self._vol(ps,bench)
            sel=self._corr(self._mom(ps),rs)
            if len(sel)==0:sel,wts=[defensive],{defensive:1.0};ew=0.0
            else:pw=ew/len(sel);wts={c:pw for c in sel}
            if ew<1.0:wts[defensive]=1.0-ew
            sigs.append(dict(date=dt,equity_weight=ew,selected=sel,weights=wts))
        return pd.DataFrame(sigs).set_index("date")

    def _rbd(self,dates):
        iso=dates.isocalendar()
        y=iso["year"] if isinstance(iso,pd.DataFrame) else iso.year
        w=iso["week"] if isinstance(iso,pd.DataFrame) else iso.week
        wid=y.values*100+w.values
        df=pd.DataFrame({"d":dates,"wid":wid})
        weekly=pd.DatetimeIndex(df.groupby("wid")["d"].last().sort_values())
        return weekly[::2]

    def _trend(self,px,b):
        if b not in px.columns:return self.bear_cap
        bp=px[b].dropna()
        if len(bp)<self.trend_ma:return self.bear_cap
        return 1.0 if bp.iloc[-1]>bp.rolling(self.trend_ma).mean().iloc[-1] else self.bear_cap

    def _vol(self,px,b):
        if b not in px.columns:return 1.0
        bp=px[b].dropna()
        if len(bp)<self.vw+1:return 1.0
        lr=np.log(bp/bp.shift(1)).dropna().iloc[-self.vw:]
        av=lr.std()*np.sqrt(252)
        return min(1.0,self.tvol/av) if av>0 else 1.0

    def _mom(self,px):
        sc=pd.Series(index=px.columns,dtype=float)
        for c in px.columns:
            p=px[c].dropna()
            if len(p)<max(self.mw):sc[c]=-np.inf;continue
            wins=[p.iloc[-1]/p.iloc[-w]-1 for w in self.mw if len(p)>=w]
            sc[c]=np.mean(wins) if wins else -np.inf
        return sc.sort_values(ascending=False)

    def _corr(self,ms,rs):
        cr=rs.iloc[-self.cw:];sel=[]
        for c in ms.index:
            if ms[c]==-np.inf or c not in cr.columns:continue
            skip=False
            for s in sel:
                pair=cr[[c,s]].dropna()
                if len(pair)>=20 and abs(pair.corr().iloc[0,1])>self.corr_th:
                    skip=True;break
            if not skip:sel.append(c)
            if len(sel)>=self.top_n:break
        return sel


class ValuationStrategy(ETFRotationStrategy):
    def __init__(self, pb_pct_series=None, **kw):
        super().__init__(**kw);self.pb=pb_pct_series
    def _val(self,dt):
        if self.pb is None:return 1.0
        a=self.pb[self.pb.index<=dt]
        if len(a)==0:return 1.0
        p=a.iloc[-1]
        if pd.isna(p):return 1.0
        if p<0.20:return 1.0
        elif p<0.50:return 0.90
        elif p<0.80:return 0.70
        else:return 0.50
    def generate_signals(self,px,bench="510300",defensive="511010"):
        ret=px.pct_change();rbd=self._rbd(px.index);sigs=[]
        for dt in rbd:
            try:p=px.index.get_loc(dt)
            except KeyError:continue
            if p<max(self.trend_ma,max(self.mw),self.cw):continue
            ps=px.iloc[:p+1];rs=ret.iloc[:p+1]
            ew=self._trend(ps,bench)*self._vol(ps,bench)*self._val(dt)
            sel=self._corr(self._mom(ps),rs)
            if len(sel)==0:sel,wts=[defensive],{defensive:1.0};ew=0.0
            else:pw=ew/len(sel);wts={c:pw for c in sel}
            if ew<1.0:wts[defensive]=1.0-ew
            sigs.append(dict(date=dt,equity_weight=ew,selected=sel,weights=wts))
        return pd.DataFrame(sigs).set_index("date")


class BacktestEngine:
    def __init__(self,ic=100_000,comm=0.00005,slip=0.0005):
        self.ic=ic;self.comm=comm;self.slip=slip
    def run(self,px,sig):
        px,sig=px.copy(),sig.copy();cash=self.ic;hld={};nvs=[];trs=[]
        for date in px.index:
            if date in sig.index:
                t,cash=self._rb(date,sig.loc[date,"weights"],hld,cash,px.loc[date].to_dict())
                if t:trs.extend(t)
            hv=sum(sh*px.loc[date].get(c,0) for c,sh in hld.items()
                   if np.isfinite(px.loc[date].get(c,np.nan)))
            nvs.append({"date":date,"nav":cash+hv})
        nav=pd.DataFrame(nvs).set_index("date");nav["returns"]=nav["nav"].pct_change()
        return {"nav":nav,"trades":pd.DataFrame(trs) if trs else pd.DataFrame(),
                "metrics":self._m(nav,px)}
    def _rb(self,date,tw,hld,cash,prices):
        tr=[]
        if not tw:return tr,cash
        cn=cash+sum(sh*prices.get(c,0) for c,sh in hld.items())
        th={c:cn*w for c,w in tw.items() if w>0}
        for c in list(hld.keys()):
            if c not in th:
                sh=hld.pop(c);p=prices.get(c,np.nan)
                if np.isfinite(p) and p>0 and sh>0:
                    ep=p*(1-self.slip);pro=sh*ep;co=sh*p*self.comm
                    cash+=pro-co;tr.append(dict(date=date,code=c,action="sell",shares=sh,price=ep,proceeds=pro,cost=co))
        for c,tv in th.items():
            p=prices.get(c,np.nan)
            if not np.isfinite(p) or p<=0:continue
            cv=hld.get(c,0)*p;diff=tv-cv
            if abs(diff)<cn*0.005:continue
            if diff>0:
                ep=p*(1+self.slip);bv=min(diff,cash);nb=int(bv/ep)
                if nb>0:
                    co=nb*ep;com=co*self.comm;cash-=co+com
                    hld[c]=hld.get(c,0)+nb;tr.append(dict(date=date,code=c,action="buy",shares=nb,price=ep,proceeds=-co,cost=com))
            else:
                if not np.isfinite(p) or p <= 0: continue
                ep=p*(1-self.slip);ns=min(int(abs(diff)/p),hld.get(c,0))
                if ns>0:
                    pro=ns*ep;com=pro*self.comm;cash+=pro-com
                    hld[c]-=ns
                    if hld[c]==0:del hld[c]
                    tr.append(dict(date=date,code=c,action="sell",shares=ns,price=ep,proceeds=pro,cost=com))
        return tr,cash
    def _m(self,nav,px):
        rets=nav["returns"].dropna()
        if len(rets)<20:return {"error":"data too short"}
        yrs=max((nav.index[-1]-nav.index[0]).days/365.25,0.5)
        tr=nav["nav"].iloc[-1]/nav["nav"].iloc[0]-1
        cagr=(1+tr)**(1/yrs)-1;av=rets.std()*np.sqrt(252)
        sh=(cagr-0.02)/av if av>0 else 0
        dd=((nav["nav"]-nav["nav"].cummax())/nav["nav"].cummax()).min()
        cm=cagr/abs(dd) if dd!=0 else 0
        mo=nav["returns"].resample("ME").apply(lambda x:(1+x).prod()-1)
        wr=(mo>0).mean();aw=mo[mo>0].mean();al=mo[mo<0].mean()
        pf=abs(aw*(mo>0).sum()/(al*(mo<0).sum())) if al!=0 and (mo<0).sum()>0 else np.inf
        bi={}
        if "510300" in px.columns:
            b=px["510300"].reindex(nav.index).ffill();br=b.pct_change().dropna()
            bt=b.iloc[-1]/b.iloc[0]-1;bc=(1+bt)**(1/yrs)-1
            te=(rets-br.reindex(rets.index)).dropna().std()*np.sqrt(252)
            bi=dict(bench_cagr=bc,bench_vol=br.std()*np.sqrt(252),
                    bench_max_dd=((b-b.cummax())/b.cummax()).min(),
                    excess_cagr=cagr-bc,tracking_err=te)
            bi["ir"]=bi["excess_cagr"]/te if te>0 else 0
        return dict(total_return=tr,cagr=cagr,annual_vol=av,sharpe=sh,
                    max_drawdown=dd,calmar=cm,win_rate_monthly=wr,
                    avg_win_monthly=aw,avg_loss_monthly=al,profit_factor=pf,years=yrs,**bi)


# ============================================================
#  4. 对比回测
# ============================================================
DEFAULT = dict(mom_windows=(60,120), top_n=5, trend_ma=200, bear_equity_cap=0.30,
               target_vol=0.15, vol_window=20, corr_threshold=0.80, corr_window=60, rebalance_freq="2W")

pb_series = pb_df["pb_pct"]

print("\n回测...")
# A: 17只 原始
sA = ETFRotationStrategy(**DEFAULT)
sigA = sA.generate_signals(px17)
rA = BacktestEngine(100_000).run(px17, sigA)

# B: 17只 估值
sB = ValuationStrategy(pb_pct_series=pb_series, **DEFAULT)
sigB = sB.generate_signals(px17)
rB = BacktestEngine(100_000).run(px17, sigB)

# C: 20只 原始
sC = ETFRotationStrategy(**DEFAULT)
sigC = sC.generate_signals(px20)
rC = BacktestEngine(100_000).run(px20, sigC)

# D: 20只 估值
sD = ValuationStrategy(pb_pct_series=pb_series, **DEFAULT)
sigD = sD.generate_signals(px20)
rD = BacktestEngine(100_000).run(px20, sigD)

# ============================================================
#  5. 输出
# ============================================================
print("\n" + "=" * 95)
print("  扩池对比")
print("=" * 95)
compare = OrderedDict([
    ("A. 17只 原始", (rA["metrics"], rA["trades"])),
    ("B. 17只 估值", (rB["metrics"], rB["trades"])),
    ("C. 20只 原始", (rC["metrics"], rC["trades"])),
    ("D. 20只 估值 (推荐)", (rD["metrics"], rD["trades"])),
])

rows = [
    ("年化收益","cagr"),("年化波动","annual_vol"),("夏普","sharpe"),
    ("最大回撤","max_drawdown"),("Calmar","calmar"),
    ("月胜率","win_rate_monthly"),("超额年化","excess_cagr"),("IR","ir"),
]
hdr = f"{'指标':<12s}"
for name in compare: hdr += f"  {name:<20s}"
print(hdr); print("-" * len(hdr))
for lb,ky in rows:
    line = f"{lb:<12s}"
    for m,_ in compare.values():
        v = m.get(ky)
        if v is None or (isinstance(v,float) and np.isnan(v)): line += f"  {'N/A':>20s}"
        elif ky in ("sharpe","calmar","ir"): line += f"  {v:>20.2f}"
        else: line += f"  {v*100:>19.2f}%"
    print(line)

print(f"{'交易次数':<12s}", end="")
for _,t in compare.values(): print(f"  {len(t):>20d}", end="")
print()
print(f"{'交易成本':<12s}", end="")
for _,t in compare.values(): print(f"  {t['cost'].sum():>18.0f} RMB", end="")
print()
print("=" * 95)

# 边际贡献
exA = rA["metrics"].get("excess_cagr",0) or 0
exB = rB["metrics"].get("excess_cagr",0) or 0
exC = rC["metrics"].get("excess_cagr",0) or 0
exD = rD["metrics"].get("excess_cagr",0) or 0

print(f"\n超额年化: {exA*100:+.2f}% → {exB*100:+.2f}% → {exC*100:+.2f}% → {exD*100:+.2f}%")
print(f"估值贡献 (17池): {(exB-exA)*100:+.2f}pp")
print(f"扩池贡献 (原始): {(exC-exA)*100:+.2f}pp")
print(f"估值贡献 (20池): {(exD-exC)*100:+.2f}pp")
print(f"总改进: {(exD-exA)*100:+.2f}pp")

ddA = rA["metrics"]["max_drawdown"]
ddD = rD["metrics"]["max_drawdown"]
print(f"回撤: {ddA*100:.1f}% → {ddD*100:.1f}%")

shA = rA["metrics"]["sharpe"]
shD = rD["metrics"]["sharpe"]
print(f"夏普: {shA:.2f} → {shD:.2f}")

# ============================================================
#  6. D版信号分析
# ============================================================
print("\n" + "=" * 95)
print("  D版 (20只+估值) 持仓分析")
print("=" * 95)
all_names = {**POOL_17, "510880":"红利ETF","159985":"豆粕ETF","512890":"红利低波ETF"}
sigD_yr = sigD.copy(); sigD_yr["year"] = sigD_yr.index.year
print(f"\n{'Year':<6s}  {'仓位':>8s}  {'选中':>50s}")
print("-" * 70)
for yr in sorted(sigD_yr["year"].unique()):
    yr_data = sigD_yr[sigD_yr["year"]==yr]
    w = yr_data["equity_weight"].mean()
    # Count how often each ETF was selected
    from collections import Counter
    cnt = Counter()
    for _, row in yr_data.iterrows():
        for c in row["selected"]:
            cnt[all_names.get(c,c)] += 1
    top3 = cnt.most_common(3)
    tops = ", ".join([f"{n}({c}次)" for n,c in top3])
    print(f"{yr:<6d}  {w*100:7.0f}%  {tops}")

# 新ETF被选中的频率
print(f"\n新ETF被选频率:")
for code in ["510880","159985","512890"]:
    count = sum(1 for _, row in sigD.iterrows() if code in row["selected"])
    print(f"  {all_names[code]}: {count}/{len(sigD)} ({count/len(sigD)*100:.0f}%)")

print("\nDone.")
