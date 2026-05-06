#!/usr/bin/env python3
"""
特征法医排查脚本 (Data Forensics Audit)
验证截面归一化 (Cross-sectional Normalization) 实现是否正确

检查：对于每个时间步，跨股票的每个特征，其截面均值应≈0，截面标准差应≈1
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'src'))
from train import RankingDataset, collate_fn, split_train_val_by_last_month, preprocess_data
from config import config

def audit_cross_norm():
    print("=" * 70)
    print("🔍 截面归一化法医审计 (Cross-Sectional Normalization Audit)")
    print("=" * 70)
    
    # 加载真实数据
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    data_file = os.path.join(data_path, 'train.csv')
    
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    print(f"📂 加载数据: {data_file}")
    full_df = pd.read_csv(data_file)
    
    # 建立股票映射
    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)
    print(f"📊 股票数量: {num_stocks}")
    
    # 数据预处理
    train_df, val_df, val_start = split_train_val_by_last_month(full_df, config['sequence_length'])
    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    
    # 标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features)
    train_data[features] = scaler.fit_transform(train_data[features])
    
    print(f"📐 特征数量: {len(features)}")
    
    # 创建数据集
    from utils import create_ranking_dataset_vectorized
    train_sequences, train_targets, train_relevance, train_stock_indices = create_ranking_dataset_vectorized(
        train_data, features, config['sequence_length']
    )
    
    dataset = RankingDataset(train_sequences, train_targets, train_relevance, train_stock_indices)
    dataloader = DataLoader(
        dataset, 
        batch_size=4,  # 用小batch以便观察
        shuffle=False,  # 保持顺序以便复现
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print("\n📡 获取第一个Batch...")
    batch = next(iter(dataloader))
    
    sequences = batch['sequences']  # [batch, max_stocks, seq_len, features]
    masks = batch['masks']          # [batch, max_stocks]
    
    batch_size, max_stocks, seq_len, feature_dim = sequences.shape
    print(f"   Batch shape: {sequences.shape}")
    print(f"   Batch size (dates): {batch_size}")
    print(f"   Max stocks per date: {max_stocks}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Feature dimension: {feature_dim}")
    
    # 选择 time_step = 10 进行审计
    time_step = 10
    print(f"\n🎯 审计时间步: {time_step}")
    
    # 找到batch中第一个有效日期（mask=1的股票数量最多的）
    valid_counts = masks.sum(dim=1).numpy()
    audit_batch_idx = np.argmax(valid_counts)
    num_valid = int(valid_counts[audit_batch_idx])
    print(f"   选择审计的batch索引: {audit_batch_idx} (有效股票数: {num_valid})")
    
    # 提取该时间步的截面特征矩阵 (排除padding)
    # shape: [num_valid_stocks, feature_dim]
    batch_sequences = sequences[audit_batch_idx, :num_valid, time_step, :].numpy()
    
    print(f"   截面矩阵 shape: {batch_sequences.shape} (stocks x features)")
    
    # 计算每个特征的截面统计
    cross_mean = batch_sequences.mean(axis=0)  # [feature_dim]
    cross_std = batch_sequences.std(axis=0)    # [feature_dim]
    
    # 找出关键特征的索引
    feature_names = features
    close_idx = feature_names.index('收盘') if '收盘' in feature_names else 1
    amount_idx = feature_names.index('成交额') if '成交额' in feature_names else 5
    
    print("\n" + "=" * 70)
    print("📊 截面统计报告 (Cross-Sectional Statistics)")
    print("=" * 70)
    
    print(f"\n{'特征名':<15} {'索引':<8} {'截面均值(Mean)':<20} {'截面标准差(Std)':<20} {'验证':<10}")
    print("-" * 70)
    
    # 检查收盘 (index 1)
    close_mean = cross_mean[close_idx]
    close_std = cross_std[close_idx]
    close_ok = "✅" if abs(close_mean) < 0.1 and abs(close_std - 1.0) < 0.1 else "❌"
    print(f"{'收盘':<15} {close_idx:<8} {close_mean:<20.6f} {close_std:<20.6f} {close_ok}")
    
    # 检查成交额 (index 5)
    amount_mean = cross_mean[amount_idx]
    amount_std = cross_std[amount_idx]
    amount_ok = "✅" if abs(amount_mean) < 0.1 and abs(amount_std - 1.0) < 0.1 else "❌"
    print(f"{'成交额':<15} {amount_idx:<8} {amount_mean:<20.6f} {amount_std:<20.6f} {amount_ok}")
    
    # 检查其他几个特征
    other_features = ['开盘', '最高', '最低', '成交量', '换手率', '涨跌幅']
    print("\n--- 其他特征抽查 ---")
    for feat in other_features:
        if feat in feature_names:
            idx = feature_names.index(feat)
            mean_val = cross_mean[idx]
            std_val = cross_std[idx]
            ok = "✅" if abs(mean_val) < 0.1 and abs(std_val - 1.0) < 0.1 else "❌"
            print(f"{feat:<15} {idx:<8} {mean_val:<20.6f} {std_val:<20.6f} {ok}")
    
    # 最终判定
    print("\n" + "=" * 70)
    all_close_ok = abs(close_mean) < 0.1 and abs(close_std - 1.0) < 0.1
    all_amount_ok = abs(amount_mean) < 0.1 and abs(amount_std - 1.0) < 0.1
    
    if all_close_ok and all_amount_ok:
        print("✅ 法医审计通过！截面归一化实现正确。")
        print("   - 收盘: Mean≈0, Std≈1 ✓")
        print("   - 成交额: Mean≈0, Std≈1 ✓")
    else:
        print("❌ 法医审计失败！截面归一化存在问题。")
        if not all_close_ok:
            print(f"   - 收盘: Mean={close_mean:.4f}, Std={close_std:.4f}")
        if not all_amount_ok:
            print(f"   - 成交额: Mean={amount_mean:.4f}, Std={amount_std:.4f}")
    print("=" * 70)

if __name__ == '__main__':
    audit_cross_norm()