# -*- coding: utf-8 -*-
"""拉取新 ETF 数据 — 彻底绕过代理，写入缓存"""
import os, time, json, sys

# ============================================================
#  0. 清理一切代理痕迹
# ============================================================
for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        del os.environ[k]

import urllib.request
urllib.request.getproxies = lambda: {}

# 必须在 import requests 前清理 requests 可能缓存的代理配置
import requests
# 强制清除 requests 的 session 级缓存
requests.utils.getproxies = lambda: {}
requests.utils.get_environ_proxies = lambda url: {}

import pandas as pd
import numpy as np

# ============================================================
#  1. 用最底层的方式请求（绕过所有代理逻辑）
# ============================================================
def raw_fetch(code, market='1'):
    """用 httplib 直连，完全不用 requests/urllib 的代理逻辑"""
    import http.client
    import ssl

    secid = f"{market}.{code}"
    path = (
        f"/api/qt/stock/kline/get"
        f"?fields1=f1,f2,f3,f4,f5,f6"
        f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116"
        f"&ut=7eea3edcaed734bea9cbfc24409ed989"
        f"&klt=101&fqt=1"
        f"&beg=20200101&end=20260622"
        f"&secid={secid}"
    )

    ctx = ssl.create_default_context()
    conn = http.client.HTTPSConnection("push2his.eastmoney.com", timeout=15, context=ctx)
    try:
        conn.request("GET", path, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "*/*",
        })
        resp = conn.getresponse()
        data = resp.read().decode('utf-8')
        return json.loads(data)
    finally:
        conn.close()


def save_to_cache(code, df):
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'etf_cache')
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{code}.csv")
    df.to_csv(path, encoding='utf-8-sig')
    return path


# ============================================================
#  2. 拉取
# ============================================================
TARGETS = [
    ("510880", "红利ETF", "1"),       # 上证
    ("159985", "豆粕ETF", "0"),       # 深证
    ("512890", "红利低波ETF", "1"),   # 上证
]

results = {}
for code, name, market in TARGETS:
    print(f"拉取 {code} {name}...", end=" ", flush=True)
    success = False
    for attempt in range(5):
        try:
            data = raw_fetch(code, market)
            klines = data.get('data', {}).get('klines', [])
            if not klines:
                print(f"attempt {attempt+1}: 空数据, 重试...", end=" ", flush=True)
                time.sleep(1)
                continue

            lines = [l.split(',') for l in klines]
            rows = []
            for l in lines:
                rows.append({
                    'date': l[0],
                    'open': float(l[1]),
                    'close': float(l[2]),
                    'high': float(l[3]),
                    'low': float(l[4]),
                    'volume': int(l[5]),
                    'amount': float(l[6]),
                })
            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()

            avg_amt = df['amount'].mean()
            path = save_to_cache(code, df[['open','high','low','close','volume','amount']])
            print(f"OK! {len(df)}条, {df.index[0].date()}~{df.index[-1].date()}, 日均{avg_amt/1e4:.0f}万 → {path}")
            results[code] = df
            success = True
            break
        except Exception as e:
            msg = str(e)[:80]
            if attempt < 4:
                print(f"retry...", end=" ", flush=True)
                time.sleep(1.5 * (attempt + 1))
            else:
                print(f"FAIL: {msg}")

    if not success:
        print(f"  [FAIL] {code} {name}: 5次尝试均失败")
    time.sleep(0.5)

print(f"\n成功: {len(results)}/3")
for code in results:
    print(f"  {code}: {len(results[code])} 条")
