#!/usr/bin/env python3
"""数据截面探针检测脚本"""
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('data/train.csv')
df['日期'] = pd.to_datetime(df['日期'])

# 随机抽取一个交易日
sample_date = df['日期'].sample(1).values[0]
day_df = df[df['日期'] == sample_date]

print(f'='*60)
print(f'探针检测日期: {sample_date}')
print(f'当日股票数量: {len(day_df)}')
print(f'='*60)

# 检查基础量价特征的截面分布
features_to_check = ['收盘', '成交量', '成交额', '涨跌幅']

for col in features_to_check:
    if col in day_df.columns:
        values = day_df[col].dropna()
        cv = values.std() / abs(values.mean()) if abs(values.mean()) > 1e-9 else 0
        print(f'\n【{col}】')
        print(f'  最大值: {values.max():.4f}')
        print(f'  最小值: {values.min():.4f}')
        print(f'  均值: {values.mean():.4f}')
        print(f'  标准差: {values.std():.4f}')
        print(f'  变异系数(CV): {cv:.4f}')

print(f'\n{"="*60}')
print('诊断结论:')
print('='*60)

# 计算收盘价的CV
if '收盘' in day_df.columns:
    close_cv = day_df['收盘'].dropna().std() / abs(day_df['收盘'].dropna().mean())
    if close_cv > 0.5:
        print(f'⚠️  收盘价CV={close_cv:.2f} > 0.5，截面差异巨大，数据未归一化')
    else:
        print(f'✓ 收盘价CV={close_cv:.2f}，截面差异可控')

# 计算成交量的CV
if '成交量' in day_df.columns:
    vol_cv = day_df['成交量'].dropna().std() / abs(day_df['成交量'].dropna().mean())
    if vol_cv > 1.0:
        print(f'⚠️  成交量CV={vol_cv:.2f} > 1.0，截面差异极大')
    else:
        print(f'✓ 成交量CV={vol_cv:.2f}，截面差异可控')