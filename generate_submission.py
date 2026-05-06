#!/usr/bin/env python3
"""
比赛提交通用脚本 - generate_submission.py
使用 Pairwise_Golden 模型 + T=0.01 温度参数生成预测结果
"""

import os, sys, pandas as pd, numpy as np, torch
sys.path.insert(0, '/app/code/src')
from model import StockTransformer

# ===== 配置 =====
MODEL_PATH = '/app/models/Pairwise_Golden/best_model.pth'
DATA_PATH = '/app/data/stock_data.csv'  # 训练数据（用于构建序列）
TEST_DATA_PATH = '/app/data/test.csv'   # 测试数据（用于获取最新交易日）
OUTPUT_PATH = '/app/test/results_output/submission.csv'
SEQ_LEN = 50
TEMPERATURE = 0.01  # 锁定赢家通吃策略

config_model = {
    'd_model': 256, 'nhead': 16, 'num_layers': 3,
    'dim_feedforward': 512, 'dropout': 0.3,
    'sequence_length': 50, 'feature_num': '158+39'
}

print("="*60)
print("比赛提交通用脚本 - T=0.01 赢家通吃模式")
print("="*60)

# 1. 加载模型
print("\n[1/5] 加载模型...")
model = StockTransformer(input_dim=197, config=config_model, num_stocks=300)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
print(f"    模型加载完成 (T={TEMPERATURE})")

# 2. 加载训练数据（用于构建历史序列）
print("\n[2/5] 加载数据...")
full_df = pd.read_csv(DATA_PATH)
full_df['日期'] = pd.to_datetime(full_df['日期'])
print(f"    训练数据: {full_df.shape}")

# 加载测试数据（用于获取最新交易日）
test_df = pd.read_csv(TEST_DATA_PATH)
test_df['日期'] = pd.to_datetime(test_df['日期'])
latest_test_date = test_df['日期'].max()
print(f"    测试数据: {test_df.shape}, 最新日期: {latest_test_date}")

# 3. 特征工程
print("\n[3/5] 特征工程...")
from utils import engineer_features_158plus39

processed = engineer_features_158plus39(full_df)
processed = processed.sort_values(['股票代码', '日期'])

# 【关键】只取前197列特征
all_cols = [c for c in processed.columns if c not in ['日期', '股票代码']]
FEATURE_197 = all_cols[:197]
processed[FEATURE_197] = processed[FEATURE_197].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
print(f"    特征数: {len(FEATURE_197)} (已截断)")

# 4. 获取股票列表和最新日期
stock_ids = sorted(full_df['股票代码'].unique())
print(f"    股票数量: {len(stock_ids)}")

# 找到测试日期之前的最新可用日期
available_dates = sorted(full_df['日期'].unique())
predict_date = max([d for d in available_dates if d <= latest_test_date])
print(f"    预测日期: {predict_date.strftime('%Y-%m-%d')}")

# 5. 构建股票历史数据字典
print("\n[4/5] 构建历史序列...")
stock_history_dict = {}
for stock_id in stock_ids:
    stock_data = processed[processed['股票代码'] == stock_id].sort_values('日期')
    stock_history_dict[stock_id] = stock_data

# 6. 推理预测
print("\n[5/5] 推理预测 (T=0.01)...")

seqs, stock_ids_ordered = [], []
for stock_id in stock_ids:
    stock_data = stock_history_dict.get(stock_id)
    if stock_data is None:
        continue
    # 收集预测日期之前的历史数据
    stock_before = stock_data[stock_data['日期'] < predict_date].tail(SEQ_LEN)
    if len(stock_before) == SEQ_LEN:
        seqs.append(stock_before[FEATURE_197].values.astype(np.float32))
        stock_ids_ordered.append(stock_id)

if len(seqs) == 0:
    raise ValueError("没有足够的股票历史数据")

seqs = np.asarray(seqs, dtype=np.float32)
print(f"    有效股票数: {len(seqs)}, 特征维度: {seqs.shape[2]}")

# 模型推理
with torch.no_grad():
    x = torch.from_numpy(seqs).unsqueeze(0)
    all_scores = model(x).squeeze(0).detach().cpu().numpy()

# 获取Top5
order = np.argsort(all_scores)[::-1]
top5_idx = order[:5]
top5_stocks = [stock_ids_ordered[i] for i in top5_idx]
top5_scores = all_scores[top5_idx]

print(f"    Top5 股票分数: {top5_scores}")

# 7. 计算T=0.01的权重
logits = top5_scores / TEMPERATURE
weights = np.exp(logits - np.max(logits))
weights = weights / weights.sum()

print(f"    Top5 权重 (T=0.01): {weights}")

# 8. 生成提交文件
print("\n[完成] 生成提交文件...")

# 保留权重 > 0.001 的股票（四舍五入到4位小数）
ROUND_DIGITS = 4
results = []
for stock_id, weight in zip(top5_stocks, weights):
    if weight > 0.001:
        rounded_weight = round(weight, ROUND_DIGITS)
        results.append({'stock_id': stock_id, 'weight': rounded_weight})

# 确保权重和 <= 1.0
total_weight = sum(r['weight'] for r in results)
if total_weight > 1.0:
    # 找到最大的权重并将其减少以确保总和 <= 1.0
    max_weight_stock = max(results, key=lambda x: x['weight'])
    max_weight_stock['weight'] = round(max_weight_stock['weight'] - (total_weight - 1.0), ROUND_DIGITS)

result_df = pd.DataFrame(results)
print(f"    输出股票数: {len(result_df)}")
print(f"    权重总和: {result_df['weight'].sum()}")

# 保存
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
result_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
print(f"    已保存: {OUTPUT_PATH}")

# 显示结果
print("\n" + "="*60)
print("提交文件预览:")
print("="*60)
print(result_df.to_string(index=False))
print("="*60)