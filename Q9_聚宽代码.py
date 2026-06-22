# ============================================================
# 聚宽回测：Q9 微盘股反转策略 (基于 Q8+Q9b 研究)
# 股票池：861520.EI 成分股全量 (401只)
# 因子：9因子等权 z-score (行业中性化)
#   Q8 三因子: reversal + momentum_2_12 + volatility
#   Q9b 六新因子: high_low_spread + ret_skew + turnover_ratio
#                 + turnover_change + extreme_ret + amihud
# 调仓：月度，Top 20%，等权
# ============================================================

import pandas as pd
import numpy as np


def initialize(context):
    set_benchmark('000852.XSHG')    # 中证1000（小盘基准，涨幅低于国证2000）
    set_option('use_real_price', True)
    log.set_level('order', 'error')

    # --- 策略参数 ---
    g.rebalance_day = 1        # 每月第1个交易日调仓
    g.top_frac = 0.20           # 选前20%
    g.cash_frac = 0.95          # 使用95%资金
    g.enable_timing = True      # 市场状态择时: 趋势市减仓, 震荡市满仓
    g.position_scale = 1.0      # 仓位比例 (1.0=满仓, 0.4=最小仓位)

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

    # --- 流通市值映射 (float_mv in 亿, 用于 turnover_ratio 计算) ---
    g.float_mv = {
        '688793.XSHG': 7.25,
        '688701.XSHG': 6.33,
        '688695.XSHG': 10.2,
        '688681.XSHG': 10.97,
        '688671.XSHG': 12.08,
        '688670.XSHG': 5.28,
        '688659.XSHG': 10.09,
        '688638.XSHG': 9.47,
        '688616.XSHG': 9.62,
        '688613.XSHG': 16.86,
        '688573.XSHG': 11.32,
        '688565.XSHG': 10.34,
        '688528.XSHG': 6.86,
        '688468.XSHG': 16.57,
        '688466.XSHG': 8.7,
        '688426.XSHG': 7.34,
        '688420.XSHG': 9.98,
        '688395.XSHG': 8.38,
        '688393.XSHG': 10.84,
        '688355.XSHG': 9.01,
        '688329.XSHG': 12.52,
        '688296.XSHG': 8.26,
        '688288.XSHG': 9.78,
        '688244.XSHG': 11.83,
        '688217.XSHG': 7.48,
        '688203.XSHG': 13.96,
        '688193.XSHG': 13.44,
        '688132.XSHG': 12.91,
        '688121.XSHG': 9.08,
        '688118.XSHG': 16.36,
        '688092.XSHG': 9.34,
        '688089.XSHG': 11.48,
        '688078.XSHG': 9.43,
        '688067.XSHG': 7.34,
        '688060.XSHG': 8.54,
        '688058.XSHG': 10.63,
        '688051.XSHG': 10.2,
        '688038.XSHG': 14.59,
        '688026.XSHG': 12.72,
        '688021.XSHG': 9.82,
        '688013.XSHG': 11.24,
        '688004.XSHG': 11.21,
        '605567.XSHG': 11.9,
        '605266.XSHG': 12.08,
        '605180.XSHG': 7.19,
        '605177.XSHG': 10.11,
        '605122.XSHG': 7.99,
        '605069.XSHG': 9.37,
        '605033.XSHG': 7.35,
        '605003.XSHG': 7.42,
        '605001.XSHG': 13.7,
        '603982.XSHG': 13.3,
        '603968.XSHG': 12.2,
        '603917.XSHG': 16.77,
        '603909.XSHG': 14.73,
        '603908.XSHG': 6.86,
        '603900.XSHG': 12.41,
        '603880.XSHG': 13.35,
        '603879.XSHG': 11.47,
        '603860.XSHG': 10.23,
        '603839.XSHG': 7.18,
        '603836.XSHG': 9.43,
        '603818.XSHG': 12.15,
        '603810.XSHG': 11.84,
        '603797.XSHG': 11.5,
        '603787.XSHG': 8.89,
        '603768.XSHG': 11.98,
        '603755.XSHG': 9.86,
        '603717.XSHG': 12.15,
        '603709.XSHG': 7.41,
        '603700.XSHG': 15.02,
        '603536.XSHG': 10.16,
        '603506.XSHG': 7.56,
        '603385.XSHG': 9.82,
        '603356.XSHG': 19.47,
        '603332.XSHG': 9.68,
        '603331.XSHG': 18.63,
        '603329.XSHG': 11.96,
        '603326.XSHG': 8.43,
        '603321.XSHG': 10.75,
        '603282.XSHG': 10.5,
        '603238.XSHG': 8.58,
        '603214.XSHG': 12.54,
        '603208.XSHG': 9.76,
        '603188.XSHG': 18.93,
        '603183.XSHG': 14.48,
        '603182.XSHG': 14.29,
        '603180.XSHG': 6.8,
        '603177.XSHG': 11.11,
        '603176.XSHG': 8.61,
        '603172.XSHG': 9.48,
        '603168.XSHG': 11.43,
        '603151.XSHG': 10.06,
        '603137.XSHG': 11.84,
        '603136.XSHG': 11.62,
        '603117.XSHG': 15.35,
        '603102.XSHG': 12.7,
        '603096.XSHG': 11.14,
        '603086.XSHG': 16.92,
        '603079.XSHG': 14.12,
        '603073.XSHG': 7.44,
        '603041.XSHG': 12.33,
        '603029.XSHG': 9.06,
        '603028.XSHG': 17.11,
        '603023.XSHG': 12.79,
        '603022.XSHG': 10.61,
        '600992.XSHG': 19.0,
        '600883.XSHG': 10.81,
        '600858.XSHG': 14.54,
        '600854.XSHG': 15.45,
        '600847.XSHG': 17.74,
        '600833.XSHG': 12.97,
        '600831.XSHG': 17.52,
        '600807.XSHG': 15.48,
        '600802.XSHG': 16.42,
        '600793.XSHG': 11.32,
        '600778.XSHG': 11.86,
        '600774.XSHG': 9.91,
        '600768.XSHG': 14.44,
        '600706.XSHG': 12.89,
        '600697.XSHG': 13.05,
        '600692.XSHG': 14.75,
        '600689.XSHG': 11.41,
        '600671.XSHG': 13.12,
        '600661.XSHG': 15.06,
        '600615.XSHG': 16.69,
        '600594.XSHG': 19.7,
        '600561.XSHG': 10.67,
        '600540.XSHG': 17.05,
        '600533.XSHG': 12.79,
        '600493.XSHG': 10.83,
        '600455.XSHG': 11.69,
        '600448.XSHG': 17.13,
        '600444.XSHG': 12.18,
        '600439.XSHG': 17.04,
        '600419.XSHG': 17.14,
        '600405.XSHG': 24.7,
        '600371.XSHG': 11.62,
        '600359.XSHG': 13.62,
        '600303.XSHG': 18.46,
        '600287.XSHG': 10.07,
        '600281.XSHG': 13.92,
        '600257.XSHG': 18.36,
        '600241.XSHG': 12.33,
        '600235.XSHG': 13.11,
        '600232.XSHG': 12.33,
        '600202.XSHG': 12.96,
        '600159.XSHG': 11.95,
        '600149.XSHG': 12.67,
        '600148.XSHG': 13.36,
        '600137.XSHG': 8.62,
        '600128.XSHG': 16.59,
        '600099.XSHG': 11.39,
        '600097.XSHG': 12.04,
        '600051.XSHG': 14.97,
        '301601.XSHE': 12.03,
        '301578.XSHE': 6.82,
        '301539.XSHE': 8.54,
        '301520.XSHE': 9.09,
        '301519.XSHE': 7.59,
        '301515.XSHE': 11.63,
        '301505.XSHE': 14.05,
        '301503.XSHE': 9.26,
        '301429.XSHE': 6.19,
        '301390.XSHE': 7.02,
        '301372.XSHE': 14.33,
        '301359.XSHE': 8.59,
        '301355.XSHE': 11.82,
        '301353.XSHE': 9.23,
        '301336.XSHE': 13.65,
        '301331.XSHE': 9.37,
        '301300.XSHE': 8.78,
        '301298.XSHE': 11.15,
        '301288.XSHE': 7.74,
        '301287.XSHE': 6.97,
        '301272.XSHE': 10.62,
        '301258.XSHE': 8.82,
        '301229.XSHE': 14.55,
        '301198.XSHE': 12.04,
        '301192.XSHE': 8.55,
        '301170.XSHE': 9.28,
        '301167.XSHE': 10.54,
        '301166.XSHE': 11.17,
        '301163.XSHE': 9.55,
        '301156.XSHE': 8.71,
        '301135.XSHE': 13.41,
        '301131.XSHE': 13.32,
        '301130.XSHE': 11.43,
        '301126.XSHE': 11.41,
        '301113.XSHE': 5.61,
        '301105.XSHE': 9.59,
        '301098.XSHE': 13.69,
        '301065.XSHE': 9.79,
        '301063.XSHE': 11.36,
        '301052.XSHE': 19.61,
        '301049.XSHE': 5.4,
        '301037.XSHE': 5.37,
        '301036.XSHE': 11.3,
        '301011.XSHE': 14.3,
        '301010.XSHE': 7.44,
        '301009.XSHE': 9.74,
        '301006.XSHE': 7.49,
        '301001.XSHE': 7.58,
        '300992.XSHE': 14.47,
        '300987.XSHE': 13.94,
        '300971.XSHE': 16.55,
        '300961.XSHE': 13.44,
        '300960.XSHE': 7.21,
        '300958.XSHE': 10.69,
        '300949.XSHE': 5.72,
        '300947.XSHE': 14.3,
        '300937.XSHE': 11.83,
        '300929.XSHE': 11.23,
        '300923.XSHE': 10.96,
        '300906.XSHE': 10.89,
        '300899.XSHE': 9.49,
        '300898.XSHE': 14.52,
        '300892.XSHE': 8.26,
        '300886.XSHE': 8.08,
        '300883.XSHE': 17.49,
        '300865.XSHE': 11.63,
        '300851.XSHE': 10.3,
        '300844.XSHE': 6.47,
        '300838.XSHE': 9.87,
        '300823.XSHE': 10.13,
        '300813.XSHE': 9.49,
        '300800.XSHE': 13.94,
        '300796.XSHE': 19.59,
        '300778.XSHE': 10.03,
        '300743.XSHE': 14.43,
        '300732.XSHE': 13.52,
        '300717.XSHE': 8.63,
        '300713.XSHE': 8.86,
        '300707.XSHE': 15.08,
        '300694.XSHE': 17.31,
        '300675.XSHE': 11.28,
        '300670.XSHE': 11.05,
        '300665.XSHE': 18.0,
        '300645.XSHE': 12.64,
        '300642.XSHE': 16.35,
        '300640.XSHE': 12.43,
        '300637.XSHE': 16.91,
        '300635.XSHE': 14.88,
        '300621.XSHE': 8.48,
        '300615.XSHE': 12.52,
        '300614.XSHE': 10.48,
        '300612.XSHE': 17.43,
        '300610.XSHE': 17.79,
        '300605.XSHE': 11.64,
        '300597.XSHE': 21.26,
        '300583.XSHE': 13.04,
        '300564.XSHE': 9.86,
        '300556.XSHE': 23.43,
        '300549.XSHE': 11.47,
        '300543.XSHE': 19.36,
        '300535.XSHE': 9.34,
        '300534.XSHE': 15.56,
        '300519.XSHE': 9.94,
        '300517.XSHE': 9.76,
        '300514.XSHE': 11.8,
        '300513.XSHE': 22.96,
        '300500.XSHE': 10.05,
        '300426.XSHE': 17.28,
        '300417.XSHE': 8.91,
        '300412.XSHE': 16.57,
        '300405.XSHE': 11.29,
        '300387.XSHE': 14.51,
        '300371.XSHE': 10.49,
        '300359.XSHE': 16.63,
        '300350.XSHE': 18.75,
        '300268.XSHE': 11.58,
        '300254.XSHE': 22.74,
        '300240.XSHE': 12.02,
        '300220.XSHE': 10.02,
        '300195.XSHE': 15.87,
        '300176.XSHE': 16.02,
        '300175.XSHE': 18.79,
        '300169.XSHE': 17.38,
        '300155.XSHE': 12.01,
        '300150.XSHE': 19.38,
        '300126.XSHE': 9.33,
        '300112.XSHE': 15.16,
        '300106.XSHE': 14.05,
        '300074.XSHE': 20.23,
        '300030.XSHE': 13.46,
        '300025.XSHE': 23.0,
        '300013.XSHE': 18.36,
        '003042.XSHE': 9.28,
        '003032.XSHE': 12.29,
        '003023.XSHE': 11.49,
        '003017.XSHE': 15.29,
        '003011.XSHE': 12.74,
        '003008.XSHE': 9.4,
        '003003.XSHE': 11.4,
        '002999.XSHE': 13.75,
        '002982.XSHE': 11.64,
        '002968.XSHE': 10.14,
        '002949.XSHE': 10.59,
        '002942.XSHE': 7.16,
        '002910.XSHE': 11.37,
        '002909.XSHE': 12.38,
        '002873.XSHE': 13.24,
        '002862.XSHE': 16.51,
        '002858.XSHE': 16.26,
        '002857.XSHE': 15.73,
        '002848.XSHE': 18.71,
        '002836.XSHE': 9.32,
        '002828.XSHE': 18.61,
        '002820.XSHE': 12.46,
        '002817.XSHE': 11.07,
        '002813.XSHE': 9.83,
        '002809.XSHE': 11.65,
        '002802.XSHE': 10.6,
        '002800.XSHE': 11.02,
        '002799.XSHE': 12.67,
        '002790.XSHE': 9.56,
        '002780.XSHE': 17.16,
        '002778.XSHE': 15.06,
        '002760.XSHE': 15.0,
        '002743.XSHE': 12.83,
        '002742.XSHE': 14.69,
        '002732.XSHE': 10.42,
        '002715.XSHE': 13.5,
        '002712.XSHE': 17.14,
        '002696.XSHE': 11.06,
        '002687.XSHE': 12.6,
        '002679.XSHE': 7.86,
        '002671.XSHE': 13.25,
        '002661.XSHE': 14.55,
        '002659.XSHE': 19.14,
        '002652.XSHE': 19.05,
        '002633.XSHE': 12.14,
        '002629.XSHE': 19.73,
        '002622.XSHE': 15.19,
        '002591.XSHE': 14.54,
        '002574.XSHE': 12.01,
        '002566.XSHE': 14.92,
        '002551.XSHE': 16.69,
        '002535.XSHE': 13.19,
        '002529.XSHE': 20.23,
        '002524.XSHE': 14.02,
        '002513.XSHE': 13.57,
        '002495.XSHE': 14.28,
        '002494.XSHE': 12.41,
        '002492.XSHE': 13.68,
        '002486.XSHE': 15.76,
        '002420.XSHE': 17.9,
        '002381.XSHE': 13.21,
        '002343.XSHE': 17.96,
        '002330.XSHE': 14.03,
        '002329.XSHE': 19.18,
        '002319.XSHE': 18.48,
        '002316.XSHE': 17.3,
        '002295.XSHE': 17.0,
        '002247.XSHE': 13.46,
        '002234.XSHE': 16.37,
        '002209.XSHE': 15.75,
        '002205.XSHE': 14.32,
        '002188.XSHE': 12.11,
        '002172.XSHE': 16.08,
        '002144.XSHE': 15.0,
        '002133.XSHE': 13.87,
        '002114.XSHE': 19.38,
        '002105.XSHE': 11.43,
        '002098.XSHE': 12.19,
        '002084.XSHE': 15.65,
        '002069.XSHE': 13.23,
        '001387.XSHE': 11.53,
        '001373.XSHE': 12.27,
        '001366.XSHE': 7.76,
        '001336.XSHE': 7.62,
        '001278.XSHE': 10.46,
        '001277.XSHE': 8.91,
        '001260.XSHE': 5.96,
        '001255.XSHE': 9.74,
        '001234.XSHE': 7.37,
        '001231.XSHE': 7.82,
        '001219.XSHE': 14.42,
        '001209.XSHE': 9.3,
        '001202.XSHE': 9.45,
        '000995.XSHE': 12.56,
        '000985.XSHE': 7.87,
        '000953.XSHE': 14.25,
        '000952.XSHE': 16.53,
        '000929.XSHE': 14.5,
        '000856.XSHE': 14.13,
        '000790.XSHE': 23.51,
        '000757.XSHE': 13.65,
        '000705.XSHE': 16.87,
        '000702.XSHE': 10.96,
        '000692.XSHE': 18.24,
        '000663.XSHE': 12.68,
        '000637.XSHE': 11.73,
        '000633.XSHE': 15.93,
        '000619.XSHE': 12.71,
        '000605.XSHE': 12.02,
        '000590.XSHE': 13.12,
        '000548.XSHE': 15.17,
        '000545.XSHE': 23.78,
        '000153.XSHE': 19.63,
        '000014.XSHE': 16.82,
    }

    run_monthly(my_rebalance, g.rebalance_day)


def update_market_state(context):
    """用中证1000指数过去约6个月涨幅判断市场状态。

    Q9c: 市场涨越久 → 趋势行情 → 反转下月越弱 → 降仓位
    只取一只指数，get_price 不会超限失败。
    """
    if not g.enable_timing:
        g.position_scale = 1.0
        return

    # 取 ~130 个交易日 (~6mo) 的指数日线
    end = context.previous_date
    start = end - pd.Timedelta(days=200)
    prices = get_price('000852.XSHG', start_date=start, end_date=end,
                       fields=['close'], fq='pre', panel=False)
    if prices is None or prices.empty or len(prices) < 100:
        g.position_scale = 1.0
        return

    close = prices['close'].astype(float)
    # 简单: 最新收盘 / 130天前收盘 - 1 ≈ 6月累计收益
    cum_6m = close.iloc[-1] / close.iloc[-min(130, len(close))] - 1

    if cum_6m <= 0.10:
        g.position_scale = 1.0
    elif cum_6m <= 0.25:
        g.position_scale = 0.6
    else:
        g.position_scale = 0.4

    log.info(f'[择时] cum_6m={cum_6m*100:.1f}% -> pos={g.position_scale:.0%}')


def my_rebalance(context):
    """月度调仓入口"""
    update_market_state(context)

    current_data = get_current_data()

    active = [s for s in g.stock_pool
              if not current_data[s].paused
              and current_data[s].day_open > 0]
    if len(active) < 50:
        return

    df = compute_factors(active, context)
    if df is None or df.empty:
        return
    df = calc_score(df)

    n_pick = max(10, int(len(df) * g.top_frac))
    top_stocks = df.nlargest(n_pick, 'score')['code'].tolist()

    do_rebalance(context, top_stocks, current_data)


def compute_factors(stocks, context):
    """计算 9 因子 + 行业中性化"""
    end = context.previous_date
    start = end - pd.Timedelta(days=400)  # ~270 trading days

    prices = get_price(stocks, start_date=start, end_date=end,
                       fields=['close','high','low','volume'],
                       fq='pre', panel=False)
    if prices is None or prices.empty:
        return None

    # Pivot each field separately
    def _pivot(field):
        df = prices.pivot_table(index='time', columns='code',
                                values=field, aggfunc='last')
        return df.dropna(axis=1, thresh=200)

    close = _pivot('close')
    high  = _pivot('high')
    low   = _pivot('low')
    volume = _pivot('volume')
    if close.empty:
        return None

    # Align all to close index
    high = high.reindex(close.index, columns=close.columns)
    low = low.reindex(close.index, columns=close.columns)
    volume = volume.reindex(close.index, columns=close.columns)

    returns = close.pct_change().dropna(how='all')
    stock_list = close.columns.tolist()
    df = pd.DataFrame(index=stock_list)
    df.index.name = 'code'

    # === 1. reversal: -(last month return) ===
    if len(close) >= 21:
        df['reversal'] = -(close.iloc[-1] / close.iloc[-21] - 1)

    # === 2. momentum_2_12: return from t-12 to t-2 ===
    if len(close) >= 252:
        df['momentum_2_12'] = close.iloc[-21] / close.iloc[-252] - 1

    # === 3. volatility: annualized daily std (12m) ===
    if len(returns) >= 250:
        df['volatility'] = returns.tail(250).std() * np.sqrt(252)

    # === 4. high_low_spread: mean((high-low)/close) over 1m ===
    if len(close) >= 21:
        hl = (high.tail(21) - low.tail(21)) / close.tail(21)
        df['high_low_spread'] = hl.mean()

    # === 5. extreme_ret: min daily return over 1m (most negative = strongest bounce) ===
    if len(returns) >= 21:
        df['extreme_ret'] = returns.tail(21).min()

    # === 6. ret_skew: skew of daily returns over 3m ===
    if len(returns) >= 63:
        df['ret_skew'] = returns.tail(63).skew()

    # === 7. turnover_ratio: avg daily volume*close / float_mv over 1m ===
    if len(volume) >= 21 and len(close) >= 21:
        amount = volume.tail(21) * close.tail(21)  # 元
        avg_amount = amount.mean()
        for s in stock_list:
            fv = g.float_mv.get(s)
            if fv and fv > 0:
                df.loc[s, 'turnover_ratio'] = avg_amount[s] / (fv * 1e8)

    # === 8. turnover_change: turnover_1m / turnover_3m_earlier - 1 ===
    if len(volume) >= 63 and len(close) >= 63:
        amt_1m = (volume.tail(21) * close.tail(21)).mean()
        amt_3m = (volume.tail(63).head(42) * close.tail(63).head(42)).mean()
        for s in stock_list:
            if amt_3m[s] > 0:
                df.loc[s, 'turnover_change'] = amt_1m[s] / amt_3m[s] - 1

    # === 9. amihud: mean(|ret| / (volume*close)) over 3m ===
    ret_3m = returns.tail(63)
    amt_3m = (volume.tail(63) * close.tail(63)).reindex(ret_3m.index, columns=ret_3m.columns)
    # Only compute where amount > 0 (NaN otherwise → skipped by mean())
    amt_safe = amt_3m.where(amt_3m > 0)
    amihud_daily = np.abs(ret_3m) / (amt_safe / 1e4)
    amihud_daily.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['amihud'] = amihud_daily.mean()

    # Drop rows missing core factor
    df = df.dropna(subset=['reversal'])
    if df.empty:
        return None

    # === 行业中性化 (7大类减均值) ===
    all_factors = ['reversal','momentum_2_12','volatility',
                   'high_low_spread','extreme_ret','ret_skew',
                   'turnover_ratio','turnover_change','amihud']

    for f in all_factors:
        if f not in df.columns or df[f].isna().all():
            continue
        df['_ind'] = df.index.map(g.stock_ind_group)
        ind_mean = df.groupby('_ind')[f].transform('mean')
        df[f'{f}_n'] = df[f] - ind_mean
        df[f'{f}_n'] = df[f'{f}_n'].fillna(df[f] - df[f].mean())
    df.drop(columns=['_ind'], inplace=True, errors='ignore')

    return df


def calc_score(df):
    """9 因子等权 z-score 打分 (方向由 Q9b Portfolio Sort 确定)"""
    factor_cols = ['reversal_n', 'momentum_2_12_n', 'volatility_n',
                   'high_low_spread_n', 'extreme_ret_n', 'ret_skew_n',
                   'turnover_ratio_n', 'turnover_change_n', 'amihud_n']
    factor_cols = [c for c in factor_cols if c in df.columns]

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

    # 等权合成: 9 因子, 方向由 Portfolio Sort spread 符号确定
    # LONG_HIGH (+): reversal_n, extreme_ret_n, amihud_n
    # LONG_LOW  (-): momentum_2_12_n, volatility_n, high_low_spread_n,
    #                 ret_skew_n, turnover_ratio_n, turnover_change_n
    sign = {
        'reversal_n': +1, 'momentum_2_12_n': -1, 'volatility_n': -1,
        'high_low_spread_n': -1, 'extreme_ret_n': +1, 'ret_skew_n': -1,
        'turnover_ratio_n': -1, 'turnover_change_n': -1, 'amihud_n': +1,
    }
    df['score'] = 0
    for col in factor_cols:
        zcol = f'z_{col}'
        if zcol in df.columns:
            df['score'] += sign.get(col, 0) * df[zcol]
    df['score'] /= len(factor_cols)

    df = df.reset_index()
    return df[['code', 'score']].dropna(subset=['score'])


def do_rebalance(context, top_stocks, current_data):
    """调仓：卖出不在 Top N 的，等权买入 Top N"""
    positions = context.portfolio.positions
    total_value = context.portfolio.portfolio_value * g.cash_frac * g.position_scale

    # 1. 卖出不在 top_stocks 的（持仓 >= 1 手才卖）
    for s in list(positions.keys()):
        if s not in top_stocks and not current_data[s].paused:
            pos = positions[s]
            min_lot = 200 if s.startswith('688') else 100
            if pos.total_amount >= min_lot:
                if s.startswith('688'):
                    order_target(s, 0, LimitOrderStyle(current_data[s].low_limit))
                else:
                    order_target(s, 0)

    # 2. 过滤买不起 1 手的 (1.2x margin)，确保 target_value 够买 2 手
    candidates = [s for s in top_stocks
                  if not current_data[s].paused and (current_data[s].last_price or 0) > 0]

    if not candidates:
        return

    while True:
        target_value = total_value / len(candidates)
        affordable = []
        for s in candidates:
            price = current_data[s].last_price
            lot = 200 if s.startswith('688') else 100
            chk = price * 1.3 * lot  # 30% margin to avoid boundary issues
            if target_value >= chk:
                affordable.append(s)

        if not affordable:
            return
        if len(affordable) == len(candidates):
            break
        candidates = affordable

    target_value = total_value / len(affordable)

    for s in affordable:
        price = current_data[s].last_price
        lot = 200 if s.startswith('688') else 100
        target_shares = int(target_value / price / lot) * lot
        if target_shares < lot:
            continue

        # 只在净变动 >= 1 手时才下单
        current_amount = positions[s].total_amount if s in positions else 0
        delta = target_shares - current_amount
        if abs(delta) >= lot:
            if s.startswith('688'):
                order_target(s, target_shares, LimitOrderStyle(price * 1.05))
            else:
                order_target(s, target_shares)


# ============================================================
# 配置说明
# ============================================================
# g.top_frac = 0.20       → Top 20% (~80只)
# g.cash_frac = 0.95      → 使用 95% 资金
#
# Q8+Q9 研究支持 (65个月, 2021-01 ~ 2026-05):
#   Q8 三因子:
#     reversal:       月均多空利差 +223bp, t=+5.36, 方向一致率 75.4%
#     momentum_2_12:  月均多空利差 -109bp, t=-2.61, 方向一致率 60.0%
#     volatility:     月均多空利差 -163bp, t=-3.53, 方向一致率 75.4%
#   Q9b 六新因子:
#     high_low_spread:月均多空利差 -213bp, t=-4.77 (LONG_LOW)
#     ret_skew:       月均多空利差 -132bp, t=-4.64 (LONG_LOW)
#     turnover_ratio: 月均多空利差 -156bp, t=-4.40 (LONG_LOW)
#     turnover_change:月均多空利差 -138bp, t=-3.61 (LONG_LOW)
#     extreme_ret:    月均多空利差 +119bp, t=+3.54 (LONG_HIGH)
#     amihud:         月均多空利差  +65bp, t=+1.93 (LONG_HIGH, marginal)
#
#   Q9c: 固定等权 > 开关式(|t|>1.5) > Bayesian Ridge → 越简单越好
#   9因子固定等权: Top年化 14.9%, Spread 25.4%, t=3.45 (出样本29月)
#
# 建议回测区间：2022-01-01 ~ 2026-06-19
# ============================================================
