#!/usr/bin/env python3
"""
为 V2 Champion 模型创建正确的 scaler
Round08 模型使用 203 维特征，但原始 scaler 只有 197 维
需要创建匹配的 scaler

关键：必须像 train.py 一样完整处理流程
1. 单股票特征工程 (engineer_features_158plus39)
2. concat 所有股票
3. 添加截面排名特征 (add_cross_sectional_ranks from utils)
4. 再 fit scaler
"""
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, '/app/code/src')
from train import feature_cloums_map
from utils import engineer_features_158plus39, add_cross_sectional_ranks
from sklearn.preprocessing import RobustScaler

print("=" * 60)
print("为 V2 Champion 创建正确维度的 Scaler")
print("=" * 60)

# 配置
SCALER_OUTPUT_PATH = '/app/model/2026-05-04_Tuning/Round08_lr3e-05_drop0.15_temp1.0_score0.1790_ep3/scaler.pkl'
DATA_PATH = '/app/data/train.csv'
SEQUENCE_LENGTH = 50

# 特征列表（与训练一致）
features = feature_cloums_map['158+39']
input_dim = len(features)
print(f"特征维度: {input_dim}")

# 加载数据
print("\n加载数据...")
df = pd.read_csv(DATA_PATH, dtype={'股票代码': str})
df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
df['日期'] = pd.to_datetime(df['日期'])
print(f"数据范围: {df['日期'].min().date()} ~ {df['日期'].max().date()}")

# 只使用最近的 2 年数据进行 fit（避免数据漂移）
cutoff_date = df['日期'].max() - pd.DateOffset(years=2)
df_fit = df[df['日期'] >= cutoff_date].copy()
print(f"使用最近 2 年数据: {cutoff_date.date()} ~ {df['日期'].max().date()}")

# 特征工程 - 逐股票处理
print("\n特征工程（逐股票）...")
groups = [group for _, group in df_fit.groupby('股票代码', sort=False)]
processed_list = []
for group in groups:
    stock_features = engineer_features_158plus39(group)
    processed_list.append(stock_features)

processed = pd.concat(processed_list).reset_index(drop=True)
print(f"单股票特征工程完成: {len(processed)} 条记录")

# 添加截面排名特征（关键！必须在 concat 后）
print("添加截面排名特征...")
processed = add_cross_sectional_ranks(processed)
print(f"添加截面排名后: {len(processed)} 条记录")

processed['日期'] = pd.to_datetime(processed['日期'])

# 检查特征是否存在
missing_features = [f for f in features if f not in processed.columns]
if missing_features:
    print(f"警告：缺少特征: {missing_features}")
else:
    print("所有特征都存在 ✓")

# 替换 inf 值
processed_clean = processed.copy()
processed_clean[features] = processed_clean[features].replace([np.inf, -np.inf], np.nan)
processed_clean = processed_clean.fillna(0)

# 提取特征矩阵
X = processed_clean[features].values
print(f"特征矩阵形状: {X.shape}")

# 训练 Scaler
print("\n训练 RobustScaler...")
scaler = RobustScaler()
scaler.fit(X)
print(f"Scaler n_features_in_: {scaler.n_features_in_}")

# 保存
print(f"\n保存 Scaler 到: {SCALER_OUTPUT_PATH}")
joblib.dump(scaler, SCALER_OUTPUT_PATH)

# 验证
print("\n验证 Scaler...")
test_data = X[:1000]
transformed = scaler.transform(test_data)
print(f"Transform 形状: {transformed.shape}")
print(f"Transform 均值 (应该接近 0): {transformed.mean(axis=0)[:5]}")
print(f"Transform 标准差 (应该接近 1): {transformed.std(axis=0)[:5]}")

print("\n" + "=" * 60)
print("✅ Scaler 创建完成")
print("=" * 60)