# ============================================================
# 聚宽回测：Q7 小微盘质量池 — 2024-2026 窗口验证
# 股票池：Q7_strategy_pool.md 排除 ⚠ 后的 40 只科技 ABC 类
# 四个版本在底部，修改 initialize 最后一行即可切换
# 回测建议区间：2024-01-01 ~ 2026-06-19（与 Q7 分析窗口一致）
# ============================================================

import pandas as pd
import numpy as np


def initialize(context):
    """初始化"""
    set_benchmark('000852.XSHG')       # 中证1000 (微盘股最接近的基准)
    set_option('use_real_price', True)
    log.set_level('order', 'error')

    # --- 股票池（硬编码，来自 Q7 分析结果）---
    g.core_pool = [
        # 医药生物 A（15只）
        '002566.XSHE', '300254.XSHE', '300519.XSHE', '300534.XSHE',
        '301065.XSHE', '301130.XSHE', '301331.XSHE', '600833.XSHG',
        '605266.XSHG', '688013.XSHG', '688067.XSHG', '688468.XSHG',
        '000153.XSHE', '002817.XSHE', '002873.XSHE',
        # 汽车 A（9只）
        '001260.XSHE', '300176.XSHE', '300694.XSHE', '301170.XSHE',
        '301192.XSHE', '301298.XSHE', '600099.XSHG', '600148.XSHG',
        '603787.XSHG',
        # 计算机 A（3只）
        '600455.XSHG', '300743.XSHE', '301503.XSHE',
        # 电力设备 A（3只）
        '688616.XSHG', '688681.XSHG', '301163.XSHE',
        # 国防军工 A（1只）
        '301359.XSHE',
    ]
    g.satellite_pool = [
        '688193.XSHG', '688613.XSHG', '000953.XSHE', '300030.XSHE',
        '688021.XSHG', '000757.XSHE',
        '300583.XSHE', '603768.XSHG', '300556.XSHE',
        '301166.XSHE', '688426.XSHG', '001373.XSHE',
    ]
    g.stock_pool = g.core_pool + g.satellite_pool  # 40 只

    # --- 策略参数 ---
    g.rebalance_day = 1         # 每月第 1 个交易日调仓
    g.stop_loss_pct = -0.25     # 单只止损线：-25%
    g.lot_size = 100

    # 切换版本：修改这一行
    #   'VERSION_A' = 简单等权（不排序，全部等手数）
    #   'VERSION_B' = 三因子排序（debt+vol+dd，不含动量）
    #   'VERSION_C' = VERSION_B + 25% 止损
    #   'VERSION_D' = 三因子 + 反转因子替换动量 + 止损
    g.version = 'VERSION_D'

    # 因子权重（VERSION_B/C 用三因子，VERSION_D 用反转因子替换动量）
    g.w_debt  = 0.40   # 资产负债率（越低越好）
    g.w_vol   = 0.35   # 年化波动率（越低越好）
    g.w_dd    = 0.25   # 最大回撤（越浅越好）
    g.w_rev   = 0.20   # 反转因子（VERSION_D 专用：近 1 月跌越多分越高）

    # 每日检查止损
    run_daily(check_stop_loss, '14:50')       # 尾盘止损
    run_monthly(my_rebalance, g.rebalance_day)


# ============================================================
# 止损检查（每天尾盘跑一次）
# ============================================================
def check_stop_loss(context):
    if g.version not in ('VERSION_C', 'VERSION_D'):
        return
    current_data = get_current_data()
    for s in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[s]
        if pos.total_amount <= 0:
            continue
        price = current_data[s].last_price
        if price <= 0 or current_data[s].paused:
            continue
        pnl = (price / pos.avg_cost) - 1
        if pnl <= g.stop_loss_pct:
            # 688 科创板必须限价单 — 下限取当日跌停价
            if s.startswith('688'):
                floor = current_data[s].low_limit
                order_target(s, 0, LimitOrderStyle(floor))
            else:
                order_target(s, 0)
            log.info(f'止损: {s} {pnl:.1%}')


# ============================================================
# 月度调仓入口
# ============================================================
def my_rebalance(context):
    current_data = get_current_data()
    pool = [s for s in g.stock_pool
            if not current_data[s].paused
            and current_data[s].day_open > 0]
    if len(pool) < 5:
        return

    if g.version == 'VERSION_A':
        # 简单等权：全部纳入，不排序
        ranked = pool
    else:
        # VERSION_B/C/D：因子打分排序
        df = get_factors(pool, context)
        if df is None or df.empty:
            return
        df = calc_score(df)
        ranked = df.sort_values('score', ascending=False)['code'].tolist()

    do_rebalance(context, ranked, current_data)


# ============================================================
# 因子计算
# ============================================================
def get_factors(stocks, context):
    end = context.previous_date
    start = end - pd.Timedelta(days=300)

    prices = get_price(stocks, start_date=start, end_date=end,
                       fields=['close'], fq='pre', panel=False)
    if prices is None or prices.empty:
        return None
    prices = prices.pivot_table(index='time', columns='code',
                                values='close', aggfunc='last')
    prices = prices.dropna(axis=1, thresh=200)
    if prices.empty:
        return None

    returns = prices.pct_change().dropna(how='all')
    stock_list = prices.columns.tolist()
    df = pd.DataFrame(index=stock_list)

    # 1. 年化波动率（越低越好）
    df['volatility'] = returns.tail(250).std() * np.sqrt(252)

    # 2. 最大回撤（越浅越好）
    df['drawdown'] = -(prices.tail(250) / prices.tail(250).cummax() - 1).min()

    # 3. 反转因子（近 1 月跌越多分越高——微盘股的均值回复特征）
    if len(prices) >= 21:
        df['reversal'] = -(prices.iloc[-1] / prices.iloc[-21] - 1)

    # 4. 动量（仅用于对比，VERSION_D 不用）
    if len(prices) >= 63:
        df['momentum'] = prices.iloc[-1] / prices.iloc[-63] - 1

    # --- 基本面 ---
    try:
        q = query(
            balance.code,
            (balance.total_liability / balance.total_assets).label('debt_ratio'),
        ).filter(balance.code.in_(stock_list))
        fund = get_fundamentals(q, date=end)
        if fund is not None and not fund.empty:
            fund = fund.set_index('code')
            df['debt_ratio'] = fund['debt_ratio'].astype(float)
    except Exception as e:
        log.warn(f'基本面获取失败: {e}')

    df.index.name = 'code'
    req = ['volatility', 'drawdown']
    if g.version == 'VERSION_D':
        req.append('reversal')
    return df.dropna(subset=req)


# ============================================================
# 打分
# ============================================================
def calc_score(df):
    # 极值截断
    for col in ['debt_ratio', 'volatility', 'drawdown', 'reversal', 'momentum']:
        if col in df.columns and not df[col].isna().all():
            lo, hi = df[col].quantile(0.05), df[col].quantile(0.95)
            df[col] = df[col].clip(lo, hi)

    # Z-score
    for col in ['debt_ratio', 'volatility', 'drawdown', 'reversal', 'momentum']:
        if col in df.columns:
            std_v = df[col].std()
            if pd.notna(std_v) and std_v > 0:
                df[f'z_{col}'] = (df[col] - df[col].mean()) / std_v
            else:
                df[f'z_{col}'] = 0

    if g.version == 'VERSION_D':
        # 反转替代动量：debt 低 + vol 低 + dd 浅 + 反转高
        df['score'] = (
            g.w_debt  * (-df.get('z_debt_ratio', 0)) +
            g.w_vol   * (-df.get('z_volatility', 0)) +
            g.w_dd    * (-df.get('z_drawdown', 0)) +
            g.w_rev   * (+df.get('z_reversal', 0))
        )
    else:
        # VERSION_B/C：三因子
        df['score'] = (
            g.w_debt  * (-df.get('z_debt_ratio', 0)) +
            g.w_vol   * (-df.get('z_volatility', 0)) +
            g.w_dd    * (-df.get('z_drawdown', 0))
        )

    df = df.reset_index()
    return df[['code', 'score']]


# ============================================================
# 调仓（所有版本共用）
# ============================================================
def do_rebalance(context, ranked_list, current_data):
    positions = context.portfolio.positions

    # 1. 清仓不在排名中的（跳过停牌）
    for s in list(positions.keys()):
        if s not in ranked_list and not current_data[s].paused:
            order_target(s, 0)

    # 2. 已在排名中的持仓，不动
    already_held = {s for s in positions if s in ranked_list and positions[s].total_amount > 0}

    # 3. 对排名中未持有的，依次买入 1 手
    cash = context.portfolio.cash
    bought = 0
    for s in ranked_list:
        if s in already_held:
            continue
        price = current_data[s].last_price
        if price is None or price <= 0:
            continue

        if s.startswith('688'):
            lot_value = price * 200
            if cash < lot_value:
                continue
            order(s, 200, LimitOrderStyle(price * 1.05))
            cash -= lot_value
        else:
            lot_value = price * 100
            if cash < lot_value:
                continue
            order(s, 100)
            cash -= lot_value

        bought += 1

    if bought > 0:
        log.info(f'[{g.version}] 持有{len(already_held)}只 新买{bought}只 现金{cash:.0f}')


# ============================================================
# 版本对照（修改 initialize 中 g.version 切换）
# ============================================================
# VERSION_A: 简单等权 — 40 只全持，不排序，不择时。验证池子的纯 alpha。
# VERSION_B: 三因子排序 — debt_ratio + volatility + drawdown。砍掉动量。
# VERSION_C: VERSION_B + 25% 尾盘止损 — 砍尾部风险。
# VERSION_D: 反转因子替换动量 + 止损 — 近1月跌越多分越高。
#
# 建议回测区间：2024-01-01 ~ 2026-06-19（与 Q7 分析窗口一致）
# ============================================================
