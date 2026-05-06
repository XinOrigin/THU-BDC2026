#!/usr/bin/env python3
"""回测启动器 - 使用Pairwise_Golden模型执行无重叠T+5回测"""
import sys
sys.path.insert(0, '/app/code/src')

# 修改配置使用Pairwise_Golden模型
from config import config
config['output_dir'] = '/app/model/Pairwise_Golden'
config['feature_num'] = '158+39'

print(f"使用模型目录: {config['output_dir']}")
print(f"特征数量: {config['feature_num']}")

# 导入并运行回测
import os
os.chdir('/app')

# 修改scaler路径
scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')

# 加载必要模块
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import multiprocessing as mp
from tqdm import tqdm

from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39

feature_cloums_map = {
    '39': [
        'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
        'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
        'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ],
    '158+39': [
        'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
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
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ]
}

feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39,
}

def preprocess_predict_data(df, stockid2idx):
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_cloums_map[config['feature_num']]
    
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    
    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc='特征工程'))
    
    processed = pd.concat(processed_list).reset_index(drop=True)
    # 确保instrument列存在（scaler期望有这个列）
    if 'instrument' not in processed.columns:
        processed['instrument'] = 0.0
    # 返回完整的特征列表
    return processed, feature_columns

def calculate_T5_band_return(stock_code, pred_date, all_data_df):
    stock_data = all_data_df[all_data_df['股票代码'] == stock_code].sort_values('日期').copy()
    stock_data['open_t1'] = stock_data.groupby('股票代码')['开盘'].shift(-1)
    stock_data['open_t5'] = stock_data.groupby('股票代码')['开盘'].shift(-5)
    pred_row = stock_data[stock_data['日期'] == pred_date]
    
    if len(pred_row) == 0:
        return None
    
    open_t1 = pred_row['open_t1'].values[0]
    open_t5 = pred_row['open_t5'].values[0]
    
    if pd.isna(open_t5):
        for shift_n in [4, 3, 2]:
            col_name = f'open_t{shift_n}'
            if col_name not in stock_data.columns:
                stock_data[col_name] = stock_data.groupby('股票代码')['开盘'].shift(-shift_n)
            open_t5 = pred_row[col_name].values[0]
            if not pd.isna(open_t5):
                break
    
    if pd.isna(open_t5) or open_t1 <= 0 or open_t5 <= 0:
        return None
    
    return (open_t5 - open_t1) / open_t1

def build_inference_sequences_fixed(data, features, sequence_length, stock_ids, latest_date):
    result = []
    stock_ids_ordered = []
    
    for stock_id in stock_ids:
        stock_data = data[data['股票代码'] == stock_id]
        stock_data = stock_data[stock_data['日期'] <= latest_date].sort_values('日期')
        
        if len(stock_data) < sequence_length:
            continue
        
        seq = stock_data.iloc[-sequence_length:][features].values
        result.append(seq)
        stock_ids_ordered.append(stock_id)
    
    if len(result) == 0:
        return None, None
    
    sequences = np.array(result)
    return sequences, stock_ids_ordered

def zero_out_instrument(sequences, features):
    """将instrument通道置零 - 但如果不存在则跳过"""
    if 'instrument' in features:
        instr_idx = features.index('instrument')
        result = sequences.copy()
        result[:, :, instr_idx] = 0
        return result
    return sequences

def predict_top5_for_date(model, sequences, stock_ids_ordered, device, temperature=1.0):
    if sequences is None or len(sequences) == 0:
        return None, None, None
    
    seq_tensor = torch.FloatTensor(sequences).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(seq_tensor)
        scores = outputs.cpu().numpy().flatten()
    
    top5_idx = np.argsort(scores)[-5:][::-1]
    top5_stocks = [stock_ids_ordered[i] for i in top5_idx]
    top5_scores = [scores[i] for i in top5_idx]
    
    # 基于原始得分计算权重
    if temperature > 0:
        base_weights = F.softmax(torch.FloatTensor(top5_scores) / temperature, dim=0).cpu().numpy()
    else:
        base_weights = np.ones(5) / 5
    
    # 动态现金仓位防御机制
    # 使用原始得分的均值作为市场温度指标
    mean_score = np.mean(top5_scores)
    
    # 防守阈值：原始得分低于此值时减少仓位
    DEFENSE_THRESHOLD = 0.0  # 可根据模型输出分布调整
    
    # 最小仓位暴露比例
    MIN_EXPOSURE = 0.3
    
    if mean_score < DEFENSE_THRESHOLD:
        # 市场信心不足，按比例缩减仓位
        # scale_factor 从 MIN_EXPOSURE (当mean_score极低时) 到 1.0 (当mean_score=DEFENSE_THRESHOLD时)
        # 使用线性插值
        scale_factor = MIN_EXPOSURE + (1.0 - MIN_EXPOSURE) * (mean_score - (-1.0)) / (DEFENSE_THRESHOLD - (-1.0))
        scale_factor = max(MIN_EXPOSURE, min(1.0, scale_factor))
        final_weights = base_weights * scale_factor
        # w_cash = 1.0 - scale_factor (隐含在该机制中)
    else:
        # 正常行情，满仓
        final_weights = base_weights
    
    return top5_stocks, final_weights, scores

# 主回测流程
print("=" * 60)
print("使用 Pairwise_Golden 模型执行无重叠 T+5 回测")
print("=" * 60)

model_path = '/app/model/Pairwise_Golden/best_model.pth'
scaler_path = '/app/model/Pairwise_Golden/scaler.pkl'

# 加载数据
full_df = pd.read_csv('/app/data/stock_data.csv', dtype={'股票代码': str})
full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
full_df['日期'] = pd.to_datetime(full_df['日期'])

# 修改日期范围：从2024-01-01开始，覆盖约2年数据
val_start = pd.Timestamp('2024-01-01')
val_end = pd.Timestamp('2026-04-24')

val_dates = sorted(full_df[(full_df['日期'] >= val_start) & (full_df['日期'] <= val_end)]['日期'].unique())
print(f"验证集日期范围: {val_start.date()} ~ {val_end.date()}")
print(f"验证集总交易日: {len(val_dates)} 天")

eval_dates = val_dates[::5]
print(f"无重叠评估日期数: {len(eval_dates)} 个")

stock_ids = sorted(full_df['股票代码'].unique())

# 处理验证集数据
val_data = full_df[(full_df['日期'] >= val_start) & (full_df['日期'] <= val_end)].copy()
processed_val, features = preprocess_predict_data(val_data, None)
processed_val[features] = processed_val[features].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

# 加载scaler
scaler = joblib.load(scaler_path)
scaler_feature_names = list(scaler.feature_names_in_)

# 确保所有scaler需要的特征都存在
for col in scaler_feature_names:
    if col not in processed_val.columns:
        processed_val[col] = 0.0

# 使用scaler期望的完整特征列表进行transform
processed_val[scaler_feature_names] = scaler.transform(processed_val[scaler_feature_names])

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

checkpoint = torch.load(model_path, map_location=device)

# 获取输入维度
input_dim = 197  # 197维特征

config_model = {
    'd_model': 256, 'nhead': 16, 'num_layers': 3, 
    'dropout': 0.3, 'dim_feedforward': 512, 'sequence_length': 50
}
num_stocks = 300

model = StockTransformer(input_dim=input_dim, config=config_model, num_stocks=num_stocks)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

print(f"模型已加载: {model_path}")

# 执行回测
temperature = 1.0
eval_results = []
all_predictions = []

print("\n开始无重叠 T+5 评估...")

for eval_date in tqdm(eval_dates, desc='T+5评估'):
    try:
        seqs, stock_ids_ordered = build_inference_sequences_fixed(
            processed_val, features, 50, stock_ids, eval_date
        )
        
        if seqs is None or len(seqs) == 0:
            continue
        
        # 将instrument置零
        seqs = zero_out_instrument(seqs, features)
        
        top5, weights, all_scores = predict_top5_for_date(
            model, seqs, stock_ids_ordered, device, temperature
        )
        
        if top5 is None:
            continue
        
        all_predictions.append(tuple(top5))
        
        # 计算T+5收益
        stock_returns = []
        valid_stocks = 0
        
        for i, stock_id in enumerate(top5):
            weight = weights[i]
            return_val = calculate_T5_band_return(stock_id, eval_date, full_df)
            
            if return_val is not None:
                stock_returns.append({
                    'stock_id': stock_id,
                    'weight': weight,
                    'return': return_val
                })
                valid_stocks += 1
        
        if valid_stocks == 0:
            continue
        
        stock_ret_df = pd.DataFrame(stock_returns)
        weighted_return = (stock_ret_df['return'] * stock_ret_df['weight']).sum()
        
        eval_results.append({
            'eval_date': eval_date,
            'predicted_stocks': top5,
            'weights': weights,
            't5_return': weighted_return,
            'num_valid_stocks': valid_stocks,
            'is_profit': 1 if weighted_return > 0 else 0
        })
        
        date_str = eval_date.strftime('%Y-%m-%d')
        pos_neg = "+" if weighted_return > 0 else ""
        print(f"{date_str}: {pos_neg}{weighted_return:.6f} | {','.join(top5[:3])}...")
        
    except Exception as e:
        print(f"评估日期 {eval_date.date()} 失败: {e}")
        import traceback
        traceback.print_exc()
        continue

# 统计结果
print("\n" + "=" * 60)
print("无重叠 T+5 回测报告 (Pairwise_Golden)")
print("=" * 60)

if len(eval_results) == 0:
    print("没有有效的评估结果")
    exit(1)

result_df = pd.DataFrame(eval_results)

total_evals = len(result_df)
profitable_evals = result_df['is_profit'].sum()
losing_evals = total_evals - profitable_evals

t5_win_rate = profitable_evals / total_evals if total_evals > 0 else 0
avg_t5_return = result_df['t5_return'].mean()
std_t5_return = result_df['t5_return'].std()
cumulative_return = (1 + result_df['t5_return']).prod() - 1

unique_predictions = len(set(all_predictions))

print(f"\n=== T+5 波段评估统计 ===")
print(f"评估区间: {val_start.date()} ~ {val_end.date()}")
print(f"无重叠评估次数: {total_evals}")
print(f"盈利次数: {profitable_evals} ({profitable_evals/total_evals*100:.1f}%)")
print(f"亏损次数: {losing_evals} ({losing_evals/total_evals*100:.1f}%)")
print(f"\nT+5 胜率: {t5_win_rate*100:.2f}%")
print(f"T+5 平均收益率: {avg_t5_return:.6f}")
print(f"T+5 收益率标准差: {std_t5_return:.6f}")
print(f"\n累计总收益率 (复利): {cumulative_return*100:.4f}%")

print(f"\n=== 特征坍塌诊断 ===")
print(f"唯一预测组合数: {unique_predictions} / {total_evals}")
if unique_predictions == 1:
    print("⚠️ 警告：所有评估使用相同股票，特征坍塌问题仍存在！")
else:
    print(f"✓ 预测有多样性，特征坍塌问题已解决！")

# 保存结果
output_path = '/app/output/pairwise_golden_backtest_results.csv'
result_df.to_csv(output_path, index=False)
print(f"\n详细结果已保存到: {output_path}")

print("\n回测完成!")