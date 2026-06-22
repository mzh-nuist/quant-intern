# ============================================================
# 聚宽回测：Q8 微盘股反转策略
# 股票池：861520.EI 成分股全量 (401只)
# 因子：reversal + volatility_n + momentum_2_12_n (行业中性化)
# 调仓：月度，Top 20%，等权
# 回测区间建议：2022-01-01 ~ 2026-06-19
# ============================================================

import pandas as pd
import numpy as np


def initialize(context):
    set_benchmark('399303.XSHE')    # 国证2000（市值1001~3000名，更贴近微盘）
    set_option('use_real_price', True)
    log.set_level('order', 'error')

    # --- 策略参数 ---
    g.rebalance_day = 1        # 每月第1个交易日调仓
    g.top_frac = 0.20           # 选前20%
    g.cash_frac = 0.95          # 使用95%资金
    g.use_momentum = True       # 是否加 momentum_2_12_n
    g.use_volatility = True     # 是否加 volatility_n

    # --- 股票池 (401只 861520 成分股) ---
    g.stock_pool = [
        '688793.XSHG', '688701.XSHG', '688695.XSHG', '688681.XSHG', '688671.XSHG', '688670.XSHG', '688659.XSHG', '688638.XSHG',
        '688616.XSHG', '688613.XSHG', '688573.XSHG', '688565.XSHG', '688528.XSHG', '688468.XSHG', '688466.XSHG', '688426.XSHG',
        '688420.XSHG', '688395.XSHG', '688393.XSHG', '688355.XSHG', '688329.XSHG', '688296.XSHG', '688288.XSHG', '688244.XSHG',
        '688217.XSHG', '688203.XSHG', '688193.XSHG', '688132.XSHG', '688121.XSHG', '688118.XSHG', '688092.XSHG', '688089.XSHG',
        '688078.XSHG', '688067.XSHG', '688060.XSHG', '688058.XSHG', '688051.XSHG', '688038.XSHG', '688026.XSHG', '688021.XSHG',
        '688013.XSHG', '688004.XSHG', '605567.XSHG', '605266.XSHG', '605180.XSHG', '605177.XSHG', '605122.XSHG', '605069.XSHG',
        '605033.XSHG', '605003.XSHG', '605001.XSHG', '603982.XSHG', '603968.XSHG', '603917.XSHG', '603909.XSHG', '603908.XSHG',
        '603900.XSHG', '603880.XSHG', '603879.XSHG', '603860.XSHG', '603839.XSHG', '603836.XSHG', '603818.XSHG', '603810.XSHG',
        '603797.XSHG', '603787.XSHG', '603768.XSHG', '603755.XSHG', '603717.XSHG', '603709.XSHG', '603700.XSHG', '603536.XSHG',
        '603506.XSHG', '603385.XSHG', '603356.XSHG', '603332.XSHG', '603331.XSHG', '603329.XSHG', '603326.XSHG', '603321.XSHG',
        '603282.XSHG', '603238.XSHG', '603214.XSHG', '603208.XSHG', '603188.XSHG', '603183.XSHG', '603182.XSHG', '603180.XSHG',
        '603177.XSHG', '603176.XSHG', '603172.XSHG', '603168.XSHG', '603151.XSHG', '603137.XSHG', '603136.XSHG', '603117.XSHG',
        '603102.XSHG', '603096.XSHG', '603086.XSHG', '603079.XSHG', '603073.XSHG', '603041.XSHG', '603029.XSHG', '603028.XSHG',
        '603023.XSHG', '603022.XSHG', '600992.XSHG', '600883.XSHG', '600858.XSHG', '600854.XSHG', '600847.XSHG', '600833.XSHG',
        '600831.XSHG', '600807.XSHG', '600802.XSHG', '600793.XSHG', '600778.XSHG', '600774.XSHG', '600768.XSHG', '600706.XSHG',
        '600697.XSHG', '600692.XSHG', '600689.XSHG', '600671.XSHG', '600661.XSHG', '600615.XSHG', '600594.XSHG', '600561.XSHG',
        '600540.XSHG', '600533.XSHG', '600493.XSHG', '600455.XSHG', '600448.XSHG', '600444.XSHG', '600439.XSHG', '600419.XSHG',
        '600405.XSHG', '600371.XSHG', '600359.XSHG', '600303.XSHG', '600287.XSHG', '600281.XSHG', '600257.XSHG', '600241.XSHG',
        '600235.XSHG', '600232.XSHG', '600202.XSHG', '600159.XSHG', '600149.XSHG', '600148.XSHG', '600137.XSHG', '600128.XSHG',
        '600099.XSHG', '600097.XSHG', '600051.XSHG', '301601.XSHE', '301578.XSHE', '301539.XSHE', '301520.XSHE', '301519.XSHE',
        '301515.XSHE', '301505.XSHE', '301503.XSHE', '301429.XSHE', '301390.XSHE', '301372.XSHE', '301359.XSHE', '301355.XSHE',
        '301353.XSHE', '301336.XSHE', '301331.XSHE', '301300.XSHE', '301298.XSHE', '301288.XSHE', '301287.XSHE', '301272.XSHE',
        '301258.XSHE', '301229.XSHE', '301198.XSHE', '301192.XSHE', '301170.XSHE', '301167.XSHE', '301166.XSHE', '301163.XSHE',
        '301156.XSHE', '301135.XSHE', '301131.XSHE', '301130.XSHE', '301126.XSHE', '301113.XSHE', '301105.XSHE', '301098.XSHE',
        '301065.XSHE', '301063.XSHE', '301052.XSHE', '301049.XSHE', '301037.XSHE', '301036.XSHE', '301011.XSHE', '301010.XSHE',
        '301009.XSHE', '301006.XSHE', '301001.XSHE', '300992.XSHE', '300987.XSHE', '300971.XSHE', '300961.XSHE', '300960.XSHE',
        '300958.XSHE', '300949.XSHE', '300947.XSHE', '300937.XSHE', '300929.XSHE', '300923.XSHE', '300906.XSHE', '300899.XSHE',
        '300898.XSHE', '300892.XSHE', '300886.XSHE', '300883.XSHE', '300865.XSHE', '300851.XSHE', '300844.XSHE', '300838.XSHE',
        '300823.XSHE', '300813.XSHE', '300800.XSHE', '300796.XSHE', '300778.XSHE', '300743.XSHE', '300732.XSHE', '300717.XSHE',
        '300713.XSHE', '300707.XSHE', '300694.XSHE', '300675.XSHE', '300670.XSHE', '300665.XSHE', '300645.XSHE', '300642.XSHE',
        '300640.XSHE', '300637.XSHE', '300635.XSHE', '300621.XSHE', '300615.XSHE', '300614.XSHE', '300612.XSHE', '300610.XSHE',
        '300605.XSHE', '300597.XSHE', '300583.XSHE', '300564.XSHE', '300556.XSHE', '300549.XSHE', '300543.XSHE', '300535.XSHE',
        '300534.XSHE', '300519.XSHE', '300517.XSHE', '300514.XSHE', '300513.XSHE', '300500.XSHE', '300426.XSHE', '300417.XSHE',
        '300412.XSHE', '300405.XSHE', '300387.XSHE', '300371.XSHE', '300359.XSHE', '300350.XSHE', '300268.XSHE', '300254.XSHE',
        '300240.XSHE', '300220.XSHE', '300195.XSHE', '300176.XSHE', '300175.XSHE', '300169.XSHE', '300155.XSHE', '300150.XSHE',
        '300126.XSHE', '300112.XSHE', '300106.XSHE', '300074.XSHE', '300030.XSHE', '300025.XSHE', '300013.XSHE', '003042.XSHE',
        '003032.XSHE', '003023.XSHE', '003017.XSHE', '003011.XSHE', '003008.XSHE', '003003.XSHE', '002999.XSHE', '002982.XSHE',
        '002968.XSHE', '002949.XSHE', '002942.XSHE', '002910.XSHE', '002909.XSHE', '002873.XSHE', '002862.XSHE', '002858.XSHE',
        '002857.XSHE', '002848.XSHE', '002836.XSHE', '002828.XSHE', '002820.XSHE', '002817.XSHE', '002813.XSHE', '002809.XSHE',
        '002802.XSHE', '002800.XSHE', '002799.XSHE', '002790.XSHE', '002780.XSHE', '002778.XSHE', '002760.XSHE', '002743.XSHE',
        '002742.XSHE', '002732.XSHE', '002715.XSHE', '002712.XSHE', '002696.XSHE', '002687.XSHE', '002679.XSHE', '002671.XSHE',
        '002661.XSHE', '002659.XSHE', '002652.XSHE', '002633.XSHE', '002629.XSHE', '002622.XSHE', '002591.XSHE', '002574.XSHE',
        '002566.XSHE', '002551.XSHE', '002535.XSHE', '002529.XSHE', '002524.XSHE', '002513.XSHE', '002495.XSHE', '002494.XSHE',
        '002492.XSHE', '002486.XSHE', '002420.XSHE', '002381.XSHE', '002343.XSHE', '002330.XSHE', '002329.XSHE', '002319.XSHE',
        '002316.XSHE', '002295.XSHE', '002247.XSHE', '002234.XSHE', '002209.XSHE', '002205.XSHE', '002188.XSHE', '002172.XSHE',
        '002144.XSHE', '002133.XSHE', '002114.XSHE', '002105.XSHE', '002098.XSHE', '002084.XSHE', '002069.XSHE', '001387.XSHE',
        '001373.XSHE', '001366.XSHE', '001336.XSHE', '001278.XSHE', '001277.XSHE', '001260.XSHE', '001255.XSHE', '001234.XSHE',
        '001231.XSHE', '001219.XSHE', '001209.XSHE', '001202.XSHE', '000995.XSHE', '000985.XSHE', '000953.XSHE', '000952.XSHE',
        '000929.XSHE', '000856.XSHE', '000790.XSHE', '000757.XSHE', '000705.XSHE', '000702.XSHE', '000692.XSHE', '000663.XSHE',
        '000637.XSHE', '000633.XSHE', '000619.XSHE', '000605.XSHE', '000590.XSHE', '000548.XSHE', '000545.XSHE', '000153.XSHE',
        '000014.XSHE',
    ]

    # --- 行业映射 (28 SW1 -> 7大类) ---
    g.stock_ind_group = {
        '688793.XSHG': "其他",
        '688701.XSHG': "其他",
        '688695.XSHG': "TMT",
        '688681.XSHG': "高端制造",
        '688671.XSHG': "其他",
        '688670.XSHG': "医药",
        '688659.XSHG': "材料",
        '688638.XSHG': "高端制造",
        '688616.XSHG': "高端制造",
        '688613.XSHG': "医药",
        '688573.XSHG': "高端制造",
        '688565.XSHG': "其他",
        '688528.XSHG': "高端制造",
        '688468.XSHG': "医药",
        '688466.XSHG': "其他",
        '688426.XSHG': "医药",
        '688420.XSHG': "高端制造",
        '688395.XSHG': "高端制造",
        '688393.XSHG': "医药",
        '688355.XSHG': "高端制造",
        '688329.XSHG': "高端制造",
        '688296.XSHG': "TMT",
        '688288.XSHG': "TMT",
        '688244.XSHG': "TMT",
        '688217.XSHG': "医药",
        '688203.XSHG': "材料",
        '688193.XSHG': "医药",
        '688132.XSHG': "其他",
        '688121.XSHG': "高端制造",
        '688118.XSHG': "TMT",
        '688092.XSHG': "高端制造",
        '688089.XSHG': "材料",
        '688078.XSHG': "TMT",
        '688067.XSHG': "医药",
        '688060.XSHG': "TMT",
        '688058.XSHG': "TMT",
        '688051.XSHG': "TMT",
        '688038.XSHG': "TMT",
        '688026.XSHG': "医药",
        '688021.XSHG': "汽车",
        '688013.XSHG': "医药",
        '688004.XSHG': "TMT",
        '605567.XSHG': "消费",
        '605266.XSHG': "医药",
        '605180.XSHG': "消费",
        '605177.XSHG': "医药",
        '605122.XSHG': "材料",
        '605069.XSHG': "其他",
        '605033.XSHG': "材料",
        '605003.XSHG': "消费",
        '605001.XSHG': "高端制造",
        '603982.XSHG': "汽车",
        '603968.XSHG': "材料",
        '603917.XSHG': "汽车",
        '603909.XSHG': "其他",
        '603908.XSHG': "消费",
        '603900.XSHG': "消费",
        '603880.XSHG': "医药",
        '603879.XSHG': "材料",
        '603860.XSHG': "其他",
        '603839.XSHG': "消费",
        '603836.XSHG': "汽车",
        '603818.XSHG': "消费",
        '603810.XSHG': "材料",
        '603797.XSHG': "其他",
        '603787.XSHG': "汽车",
        '603768.XSHG': "汽车",
        '603755.XSHG': "消费",
        '603717.XSHG': "其他",
        '603709.XSHG': "消费",
        '603700.XSHG': "高端制造",
        '603536.XSHG': "消费",
        '603506.XSHG': "其他",
        '603385.XSHG': "消费",
        '603356.XSHG': "高端制造",
        '603332.XSHG': "材料",
        '603331.XSHG': "高端制造",
        '603329.XSHG': "汽车",
        '603326.XSHG': "消费",
        '603321.XSHG': "高端制造",
        '603282.XSHG': "其他",
        '603238.XSHG': "消费",
        '603214.XSHG': "其他",
        '603208.XSHG': "消费",
        '603188.XSHG': "材料",
        '603183.XSHG': "其他",
        '603182.XSHG': "其他",
        '603180.XSHG': "消费",
        '603177.XSHG': "其他",
        '603176.XSHG': "其他",
        '603172.XSHG': "材料",
        '603168.XSHG': "医药",
        '603151.XSHG': "其他",
        '603137.XSHG': "其他",
        '603136.XSHG': "其他",
        '603117.XSHG': "汽车",
        '603102.XSHG': "消费",
        '603096.XSHG': "其他",
        '603086.XSHG': "材料",
        '603079.XSHG': "材料",
        '603073.XSHG': "材料",
        '603041.XSHG': "材料",
        '603029.XSHG': "高端制造",
        '603028.XSHG': "高端制造",
        '603023.XSHG': "汽车",
        '603022.XSHG': "消费",
        '600992.XSHG': "高端制造",
        '600883.XSHG': "其他",
        '600858.XSHG': "其他",
        '600854.XSHG': "其他",
        '600847.XSHG': "高端制造",
        '600833.XSHG': "医药",
        '600831.XSHG': "其他",
        '600807.XSHG': "医药",
        '600802.XSHG': "材料",
        '600793.XSHG': "消费",
        '600778.XSHG': "其他",
        '600774.XSHG': "医药",
        '600768.XSHG': "材料",
        '600706.XSHG': "其他",
        '600697.XSHG': "其他",
        '600692.XSHG': "其他",
        '600689.XSHG': "其他",
        '600671.XSHG': "医药",
        '600661.XSHG': "其他",
        '600615.XSHG': "材料",
        '600594.XSHG': "医药",
        '600561.XSHG': "汽车",
        '600540.XSHG': "其他",
        '600533.XSHG': "其他",
        '600493.XSHG': "消费",
        '600455.XSHG': "TMT",
        '600448.XSHG': "消费",
        '600444.XSHG': "高端制造",
        '600439.XSHG': "消费",
        '600419.XSHG': "消费",
        '600405.XSHG': "高端制造",
        '600371.XSHG': "其他",
        '600359.XSHG': "其他",
        '600303.XSHG': "汽车",
        '600287.XSHG': "其他",
        '600281.XSHG': "材料",
        '600257.XSHG': "其他",
        '600241.XSHG': "高端制造",
        '600235.XSHG': "消费",
        '600232.XSHG': "高端制造",
        '600202.XSHG': "高端制造",
        '600159.XSHG': "其他",
        '600149.XSHG': "其他",
        '600148.XSHG': "汽车",
        '600137.XSHG': "消费",
        '600128.XSHG': "其他",
        '600099.XSHG': "汽车",
        '600097.XSHG': "其他",
        '600051.XSHG': "其他",
        '301601.XSHE': "高端制造",
        '301578.XSHE': "其他",
        '301539.XSHE': "汽车",
        '301520.XSHE': "医药",
        '301519.XSHE': "其他",
        '301515.XSHE': "医药",
        '301505.XSHE': "其他",
        '301503.XSHE': "TMT",
        '301429.XSHE': "材料",
        '301390.XSHE': "其他",
        '301372.XSHE': "其他",
        '301359.XSHE': "TMT",
        '301355.XSHE': "消费",
        '301353.XSHE': "高端制造",
        '301336.XSHE': "消费",
        '301331.XSHE': "医药",
        '301300.XSHE': "材料",
        '301298.XSHE': "汽车",
        '301288.XSHE': "其他",
        '301287.XSHE': "消费",
        '301272.XSHE': "高端制造",
        '301258.XSHE': "医药",
        '301229.XSHE': "汽车",
        '301198.XSHE': "消费",
        '301192.XSHE': "汽车",
        '301170.XSHE': "汽车",
        '301167.XSHE': "其他",
        '301166.XSHE': "医药",
        '301163.XSHE': "高端制造",
        '301156.XSHE': "材料",
        '301135.XSHE': "其他",
        '301131.XSHE': "材料",
        '301130.XSHE': "医药",
        '301126.XSHE': "医药",
        '301113.XSHE': "消费",
        '301105.XSHE': "高端制造",
        '301098.XSHE': "其他",
        '301065.XSHE': "医药",
        '301063.XSHE': "高端制造",
        '301052.XSHE': "其他",
        '301049.XSHE': "其他",
        '301037.XSHE': "材料",
        '301036.XSHE': "材料",
        '301011.XSHE': "消费",
        '301010.XSHE': "材料",
        '301009.XSHE': "其他",
        '301006.XSHE': "高端制造",
        '301001.XSHE': "其他",
        '300992.XSHE': "高端制造",
        '300987.XSHE': "其他",
        '300971.XSHE': "高端制造",
        '300961.XSHE': "其他",
        '300960.XSHE': "高端制造",
        '300958.XSHE': "其他",
        '300949.XSHE': "其他",
        '300947.XSHE': "其他",
        '300937.XSHE': "医药",
        '300929.XSHE': "其他",
        '300923.XSHE': "高端制造",
        '300906.XSHE': "高端制造",
        '300899.XSHE': "其他",
        '300898.XSHE': "消费",
        '300892.XSHE': "消费",
        '300886.XSHE': "其他",
        '300883.XSHE': "消费",
        '300865.XSHE': "高端制造",
        '300851.XSHE': "高端制造",
        '300844.XSHE': "其他",
        '300838.XSHE': "高端制造",
        '300823.XSHE': "高端制造",
        '300813.XSHE': "高端制造",
        '300800.XSHE': "其他",
        '300796.XSHE': "材料",
        '300778.XSHE': "其他",
        '300743.XSHE': "TMT",
        '300732.XSHE': "其他",
        '300717.XSHE': "材料",
        '300713.XSHE': "高端制造",
        '300707.XSHE': "汽车",
        '300694.XSHE': "汽车",
        '300675.XSHE': "其他",
        '300670.XSHE': "高端制造",
        '300665.XSHE': "材料",
        '300645.XSHE': "TMT",
        '300642.XSHE': "医药",
        '300640.XSHE': "消费",
        '300637.XSHE': "材料",
        '300635.XSHE': "其他",
        '300621.XSHE': "其他",
        '300615.XSHE': "TMT",
        '300614.XSHE': "其他",
        '300612.XSHE': "其他",
        '300610.XSHE': "材料",
        '300605.XSHE': "TMT",
        '300597.XSHE': "TMT",
        '300583.XSHE': "医药",
        '300564.XSHE': "其他",
        '300556.XSHE': "TMT",
        '300549.XSHE': "高端制造",
        '300543.XSHE': "其他",
        '300535.XSHE': "材料",
        '300534.XSHE': "医药",
        '300519.XSHE': "医药",
        '300517.XSHE': "其他",
        '300514.XSHE': "高端制造",
        '300513.XSHE': "TMT",
        '300500.XSHE': "其他",
        '300426.XSHE': "其他",
        '300417.XSHE': "高端制造",
        '300412.XSHE': "高端制造",
        '300405.XSHE': "材料",
        '300387.XSHE': "材料",
        '300371.XSHE': "高端制造",
        '300359.XSHE': "其他",
        '300350.XSHE': "汽车",
        '300268.XSHE': "其他",
        '300254.XSHE': "医药",
        '300240.XSHE': "汽车",
        '300220.XSHE': "消费",
        '300195.XSHE': "高端制造",
        '300176.XSHE': "汽车",
        '300175.XSHE': "其他",
        '300169.XSHE': "材料",
        '300155.XSHE': "TMT",
        '300150.XSHE': "TMT",
        '300126.XSHE': "高端制造",
        '300112.XSHE': "高端制造",
        '300106.XSHE': "消费",
        '300074.XSHE': "TMT",
        '300030.XSHE': "医药",
        '300025.XSHE': "TMT",
        '300013.XSHE': "汽车",
        '003042.XSHE': "材料",
        '003032.XSHE': "其他",
        '003023.XSHE': "其他",
        '003017.XSHE': "材料",
        '003011.XSHE': "消费",
        '003008.XSHE': "其他",
        '003003.XSHE': "消费",
        '002999.XSHE': "材料",
        '002982.XSHE': "其他",
        '002968.XSHE': "其他",
        '002949.XSHE': "其他",
        '002942.XSHE': "材料",
        '002910.XSHE': "消费",
        '002909.XSHE': "材料",
        '002873.XSHE': "医药",
        '002862.XSHE': "消费",
        '002858.XSHE': "其他",
        '002857.XSHE': "高端制造",
        '002848.XSHE': "其他",
        '002836.XSHE': "消费",
        '002828.XSHE': "其他",
        '002820.XSHE': "消费",
        '002817.XSHE': "医药",
        '002813.XSHE': "汽车",
        '002809.XSHE': "材料",
        '002802.XSHE': "材料",
        '002800.XSHE': "汽车",
        '002799.XSHE': "消费",
        '002790.XSHE': "消费",
        '002780.XSHE': "消费",
        '002778.XSHE': "其他",
        '002760.XSHE': "高端制造",
        '002743.XSHE': "其他",
        '002742.XSHE': "医药",
        '002732.XSHE': "消费",
        '002715.XSHE': "汽车",
        '002712.XSHE': "其他",
        '002696.XSHE': "其他",
        '002687.XSHE': "消费",
        '002679.XSHE': "其他",
        '002671.XSHE': "材料",
        '002661.XSHE': "消费",
        '002659.XSHE': "其他",
        '002652.XSHE': "材料",
        '002633.XSHE': "高端制造",
        '002629.XSHE': "其他",
        '002622.XSHE': "医药",
        '002591.XSHE': "材料",
        '002574.XSHE': "消费",
        '002566.XSHE': "医药",
        '002551.XSHE': "医药",
        '002535.XSHE': "高端制造",
        '002529.XSHE': "高端制造",
        '002524.XSHE': "医药",
        '002513.XSHE': "材料",
        '002495.XSHE': "消费",
        '002494.XSHE': "消费",
        '002492.XSHE': "汽车",
        '002486.XSHE': "消费",
        '002420.XSHE': "其他",
        '002381.XSHE': "材料",
        '002343.XSHE': "其他",
        '002330.XSHE': "消费",
        '002329.XSHE': "消费",
        '002319.XSHE': "材料",
        '002316.XSHE': "其他",
        '002295.XSHE': "材料",
        '002247.XSHE': "材料",
        '002234.XSHE': "其他",
        '002209.XSHE': "高端制造",
        '002205.XSHE': "材料",
        '002188.XSHE': "其他",
        '002172.XSHE': "医药",
        '002144.XSHE': "消费",
        '002133.XSHE': "其他",
        '002114.XSHE': "材料",
        '002105.XSHE': "汽车",
        '002098.XSHE': "消费",
        '002084.XSHE': "消费",
        '002069.XSHE': "其他",
        '001387.XSHE': "其他",
        '001373.XSHE': "TMT",
        '001366.XSHE': "其他",
        '001336.XSHE': "其他",
        '001278.XSHE': "汽车",
        '001277.XSHE': "高端制造",
        '001260.XSHE': "汽车",
        '001255.XSHE': "材料",
        '001234.XSHE': "消费",
        '001231.XSHE': "材料",
        '001219.XSHE': "消费",
        '001209.XSHE': "消费",
        '001202.XSHE': "汽车",
        '000995.XSHE': "消费",
        '000985.XSHE': "其他",
        '000953.XSHE': "医药",
        '000952.XSHE': "医药",
        '000929.XSHE': "消费",
        '000856.XSHE': "高端制造",
        '000790.XSHE': "医药",
        '000757.XSHE': "汽车",
        '000705.XSHE': "医药",
        '000702.XSHE': "其他",
        '000692.XSHE': "其他",
        '000663.XSHE': "其他",
        '000637.XSHE': "其他",
        '000633.XSHE': "材料",
        '000619.XSHE': "材料",
        '000605.XSHE': "其他",
        '000590.XSHE': "医药",
        '000548.XSHE': "汽车",
        '000545.XSHE': "材料",
        '000153.XSHE': "医药",
        '000014.XSHE': "其他",
    }

    run_monthly(my_rebalance, g.rebalance_day)


def my_rebalance(context):
    """月度调仓入口"""
    current_data = get_current_data()

    # 过滤停牌
    active = [s for s in g.stock_pool
              if not current_data[s].paused
              and current_data[s].day_open > 0]
    if len(active) < 50:
        return

    # 计算因子 + 打分
    df = compute_factors(active, context)
    if df is None or df.empty:
        return
    df = calc_score(df)

    # 取 Top 20%
    n_pick = max(10, int(len(df) * g.top_frac))
    top_stocks = df.nlargest(n_pick, 'score')['code'].tolist()

    do_rebalance(context, top_stocks, current_data)


def compute_factors(stocks, context):
    """计算 3 个价格因子 + 行业中性化"""
    end = context.previous_date
    start = end - pd.Timedelta(days=400)  # ~270 trading days, need 252 for momentum

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
    df.index.name = 'code'

    # 1. reversal: -(last 1 month return)
    if len(prices) >= 21:
        df['reversal'] = -(prices.iloc[-1] / prices.iloc[-21] - 1)

    # 2. momentum_2_12: return from t-12 to t-2 (skip last month)
    if len(prices) >= 252:
        df['momentum_2_12'] = prices.iloc[-21] / prices.iloc[-252] - 1

    # 3. volatility: annualized daily std (12 months)
    if len(returns) >= 250:
        df['volatility'] = returns.tail(250).std() * np.sqrt(252)

    # Drop rows missing core factor
    df = df.dropna(subset=['reversal'])
    if df.empty:
        return None

    # --- 行业中性化 (7大类减均值) ---
    for f in ['reversal', 'momentum_2_12', 'volatility']:
        if f not in df.columns or df[f].isna().all():
            continue
        df['_ind'] = df.index.map(g.stock_ind_group)
        ind_mean = df.groupby('_ind')[f].transform('mean')
        df[f'{f}_n'] = df[f] - ind_mean
        df[f'{f}_n'] = df[f'{f}_n'].fillna(df[f] - df[f].mean())
    df.drop(columns=['_ind'], inplace=True, errors='ignore')

    return df


def calc_score(df):
    """等权 z-score 打分"""
    factor_cols = ['reversal_n']
    if g.use_momentum and 'momentum_2_12_n' in df.columns:
        factor_cols.append('momentum_2_12_n')
    if g.use_volatility and 'volatility_n' in df.columns:
        factor_cols.append('volatility_n')

    # Winsorize (1st/99th percentile)
    for col in factor_cols:
        if col in df.columns and df[col].notna().sum() > 10:
            lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)

    # Z-score
    for col in factor_cols:
        if col in df.columns:
            mu = df[col].mean()
            sigma = df[col].std()
            df[f'z_{col}'] = (df[col] - mu) / sigma if sigma > 0 else 0

    # 等权合成
    # reversal_n: + (越高越好 = 跌越多越好)
    # momentum_2_12_n: - (越低越好 = 中期输家 = 与反转同向)
    # volatility_n: - (越低越好 = 低波动 = 与反转同向)
    df['score'] = df.get('z_reversal_n', 0)

    sign_map = {
        'reversal_n': +1,
        'momentum_2_12_n': -1,
        'volatility_n': -1,
    }
    for col in factor_cols:
        if col != 'reversal_n' and f'z_{col}' in df.columns:
            df['score'] += sign_map.get(col, 0) * df[f'z_{col}']
    df['score'] /= len(factor_cols)

    df = df.reset_index()
    return df[['code', 'score']].dropna(subset=['score'])


def do_rebalance(context, top_stocks, current_data):
    """调仓：卖出不在 Top N 的，等权买入 Top N（过滤买不起的）"""
    positions = context.portfolio.positions

    # 1. 卖出不在 top_stocks 的（order_target 避免碎股平仓被拒）
    for s in list(positions.keys()):
        if s not in top_stocks and not current_data[s].paused:
            if s.startswith('688'):
                order_target(s, 0, LimitOrderStyle(current_data[s].low_limit))
            else:
                order_target(s, 0)

    # 2. 过滤买不起 1 手的股票，等权分配
    # 用 1.2x 安全边际：last_price 到开盘执行可能跳价，避免刚好卡边界
    total_cash = context.portfolio.portfolio_value * g.cash_frac
    raw_target = total_cash / len(top_stocks)

    affordable = []
    for s in top_stocks:
        price = current_data[s].last_price
        if price is None or price <= 0 or current_data[s].paused:
            continue
        min_value = price * 1.2 * (200 if s.startswith('688') else 100)
        if raw_target >= min_value:
            affordable.append(s)

    if not affordable:
        return

    # 重新等权分配
    target_value = total_cash / len(affordable)

    for s in affordable:
        price = current_data[s].last_price
        if s.startswith('688'):
            order_target_value(s, target_value, LimitOrderStyle(price * 1.05))
        else:
            order_target_value(s, target_value)


# ============================================================
# 配置说明
# ============================================================
# g.use_momentum = True   → 启用 momentum_2_12_n (推荐)
# g.use_volatility = True → 启用 volatility_n (推荐)
# g.top_frac = 0.20       → Top 20% (~80只)
# g.cash_frac = 0.95      → 使用 95% 资金
#
# Q8 研究支持 (65个月, 2021-01 ~ 2026-05):
#   reversal:      月均多空利差 +223bp, t=+5.36, 方向一致率 75.4%
#   volatility_n:  月均多空利差 -163bp, t=-3.53, 方向一致率 75.4%
#   momentum_2_12: 月均多空利差 -109bp, t=-2.61, 方向一致率 60.0%
#   debt_ratio:    月均多空利差   +4bp, t=+0.16 (无预测力 — 剔除)
#   drawdown:      月均多空利差   -6bp, t=-0.12 (无预测力 — 剔除)
#
# 建议回测区间：2022-01-01 ~ 2026-06-19
# ============================================================
