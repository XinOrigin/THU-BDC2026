#!/usr/bin/env python3
"""
训练完成后自动执行无重叠 T+5 回测
"""
import os
import time
import subprocess

model_dir = "/app/model/Golden_NoInstrument_20260502_114017"
final_score_path = os.path.join(model_dir, "final_score.txt")
max_wait = 600  # 最多等10分钟

print("等待训练完成...")
start = time.time()
while not os.path.exists(final_score_path):
    if time.time() - start > max_wait:
        print("超时！训练未完成")
        exit(1)
    time.sleep(10)
    print(f"已等待 {int(time.time()-start)} 秒...")

print("训练完成！")

# 读取训练结果
with open(final_score_path, 'r') as f:
    content = f.read()
print(f"训练结果:\n{content}")

# 查找模型文件
model_path = None
for f in os.listdir(model_dir):
    if f.startswith('best_model') and f.endswith('.pth'):
        model_path = os.path.join(model_dir, f)
        break

if model_path is None:
    print("错误：未找到模型文件")
    exit(1)

print(f"模型路径: {model_path}")

# 保存模型路径
with open('/app/output/golden_model_path.txt', 'w') as f:
    f.write(f"{model_dir}\n{model_path}\n")

print("\n开始执行无重叠 T+5 回测...")

# 创建回测脚本
backtest_script = """
import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, '/app/code/src')
from config import config
from model import StockTransformer
from utils import engineer_features_158plus39

# 设置
model_path = '{model_path}'
output_dir = '/app/output'
os.makedirs(output_dir, exist_ok=True)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(model_path, map_location=device)

# 获取特征数量（从模型权重形状推断）
state_dict = checkpoint
# 找到第一个fc或embedding层的权重形状
input_dim = None
for key in state_dict.keys():
    if 'fc_in' in key or 'embedding' in key:
        input_dim = state_dict[key].shape[0]
        break
if input_dim is None:
    # 从训练日志推断是196
    input_dim = 196
    
print(f"模型输入维度: {{input_dim}}")

config_model = {{'d_model': 256, 'nhead': 16, 'num_layers': 3, 'dropout': 0.3, 'dim_feedforward': 512}}
num_stocks = 300

model = StockTransformer(input_dim=input_dim, config=config_model, num_stocks=num_stocks)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# 加载数据
data = pd.read_csv('/app/data/stock_data.csv')
data['datetime'] = pd.to_datetime(data['日期'])
data['股票代码'] = data['股票代码'].astype(str)

# stockid2idx
stockid2idx = {{}}
for idx, code in enumerate(sorted(data['股票代码'].unique())):
    stockid2idx[code] = idx

# 加载 scaler
scaler = joblib.load('/app/model/50_158+39/scaler.pkl')

# 验证集日期范围
val_dates = sorted(data['datetime'].unique())
val_dates = [d for d in val_dates if d >= pd.Timestamp('2026-01-01') and d <= pd.Timestamp('2026-04-24')]
print(f"验证集日期范围: {{val_dates[0]}} ~ {{val_dates[-1]}}")

# 无重叠评估（每5天）
eval_dates = val_dates[::5]
print(f"无重叠评估日期数: {{len(eval_dates)}}")

# 特征列表（无 instrument）
feature_columns = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
    'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
    'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
    'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
    'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
    'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
    'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
    'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
    'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
    'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
    'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
    'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
    'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
    'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
    'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
    'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
    'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
    'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
    'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
    'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread']

def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
    result = []
    stock_ids_ordered = []
    
    for stock_id in stock_ids:
        stock_data = data[data['股票代码'] == stock_id]
        stock_data = stock_data[stock_data['datetime'] <= latest_date].sort_values('datetime')
        
        if len(stock_data) < sequence_length:
            continue
            
        seq = stock_data.iloc[-sequence_length:][features].values
        result.append(seq)
        stock_ids_ordered.append(stock_id)
        
    if len(result) == 0:
        return None, None
        
    sequences = np.array(result)
    return sequences, stock_ids_ordered

results = []
unique_predictions = set()

for eval_date in tqdm(eval_dates, desc='T+5评估'):
    try:
        stock_ids = sorted(data['股票代码'].unique())
        seqs, stock_ids_ordered = build_inference_sequences(data, feature_columns, 50, stock_ids, eval_date)
        
        if seqs is None or len(seqs) == 0:
            print(f"评估日期 {{eval_date}} 失败: 没有可用于预测的股票序列")
            continue
            
        # 标准化
        seqs_flat = seqs.reshape(-1, seqs.shape[-1])
        seqs_flat = scaler.transform(seqs_flat)
        seqs = seqs_flat.reshape(seqs.shape)
        
        # 转换为tensor
        seqs_tensor = torch.FloatTensor(seqs).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(seqs_tensor)
            scores = outputs.cpu().numpy()
        
        # 获取Top5
        top5_idx = np.argsort(scores, axis=0).flatten()[-5:][::-1]
        top5_stocks = [stock_ids_ordered[i] for i in top5_idx]
        top5_scores = [scores[i] for i in top5_idx]
        
        unique_predictions.add(tuple(sorted(top5_stocks[:3])))
        
        # 计算T+5收益
        t1_date = eval_date + pd.Timedelta(days=1)
        while t1_date not in val_dates:
            t1_date += pd.Timedelta(days=1)
        
        t5_date = t1_date + pd.Timedelta(days=5)
        while t5_date not in val_dates:
            t5_date += pd.Timedelta(days=1)
        
        returns = []
        for stock in top5_stocks:
            try:
                p_t1 = data[(data['股票代码'] == stock) & (data['datetime'] == t1_date)]['开盘'].values[0]
                p_t5 = data[(data['股票代码'] == stock) & (data['datetime'] == t5_date)]['开盘'].values[0]
                ret = (p_t5 - p_t1) / p_t1
                returns.append(ret)
            except:
                returns.append(0)
        
        weighted_return = np.mean(returns)
        
        results.append({{
            'date': str(eval_date.date()),
            'top5': top5_stocks,
            'weighted_return': weighted_return,
            'is_profit': weighted_return > 0
        }})
        
        print(f"{{eval_date.date()}}: {{'+.4f' if weighted_return >= 0 else '%.4f' % weighted_return}} | {{','.join(map(str, top5_stocks[:3]))}}...")
        
    except Exception as e:
        print(f"评估日期 {{eval_date}} 失败: {{str(e)}}")
        continue

# 统计结果
if len(results) > 0:
    profits = sum(1 for r in results if r['is_profit'])
    total = len(results)
    win_rate = profits / total * 100 if total > 0 else 0
    avg_return = np.mean([r['weighted_return'] for r in results])
    cum_return = np.prod([1 + r['weighted_return'] for r in results]) - 1
    
    print(f"\n{'='*60}")
    print(f"T+5 回测报告")
    print(f"{'='*60}")
    print(f"总有效评估次数: {{total}}")
    print(f"T+5 胜率: {{win_rate:.2f}}%")
    print(f"累计收益率 (复利): {{cum_return*100:.4f}}%")
    print(f"唯一预测组合数: {{len(unique_predictions)}}")
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(f'{{output_dir}}/golden_backtest_results.csv', index=False)
else:
    print("没有有效的评估结果")
""".format(model_path=model_path)

# 执行回测脚本
with open('/app/golden_backtest.py', 'w') as f:
    f.write(backtest_script)

result = subprocess.run(['python', '/app/golden_backtest.py'], capture_output=False)
exit(result.returncode)