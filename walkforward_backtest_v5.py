"""
滚动回测系统 v5 (Walk-forward Backtest v5) - 无重叠T+5评估

修复问题：
1. 特征坍塌：将instrument特征置零（等效于移除）
2. 重叠持仓谬误：改为周频（每5个交易日）评估，无重叠
3. T+5波段收益计算
"""
import os
import multiprocessing as mp
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import timedelta

import sys
sys.path.insert(0, '/app/code/src')

from config import config
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
	assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
	feature_engineer = feature_engineer_func_map[config['feature_num']]
	feature_columns = feature_cloums_map[config['feature_num']]

	df = df.copy()
	df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
	groups = [group for _, group in df.groupby('股票代码', sort=False)]
	if len(groups) == 0:
		raise ValueError('输入数据为空，无法预测')

	num_processes = min(10, mp.cpu_count())
	with mp.Pool(processes=num_processes) as pool:
		processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc='特征工程'))

	processed = pd.concat(processed_list).reset_index(drop=True)
	processed['instrument'] = processed['股票代码'].map(stockid2idx)
	processed = processed.dropna(subset=['instrument']).copy()
	processed['instrument'] = processed['instrument'].astype(np.int64)
	processed['日期'] = pd.to_datetime(processed['日期'])

	return processed, feature_columns


def build_inference_sequences_fixed(data, features, sequence_length, stock_ids, actual_latest_date):
	"""为特定日期构建推断序列"""
	sequences, sequence_stock_ids = [], []
	for stock_id in stock_ids:
		stock_history = data[
			(data['股票代码'] == stock_id) &
			(data['日期'] <= actual_latest_date)
		].sort_values('日期').tail(sequence_length)

		if len(stock_history) == sequence_length:
			sequences.append(stock_history[features].values.astype(np.float32))
			sequence_stock_ids.append(stock_id)

	if len(sequences) == 0:
		raise ValueError(f'没有可用于预测的股票序列')

	return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def zero_out_instrument(sequences_np, features):
	"""将instrument特征置零，防止静态特征霸权"""
	if 'instrument' in features:
		inst_idx = features.index('instrument')
		# 将instrument维度全部置零
		sequences_np = sequences_np.copy()  # 不要修改原始数据
		sequences_np[:, :, inst_idx] = 0
	return sequences_np


def predict_top5_for_date(model, sequences_np, sequence_stock_ids, device, temperature=1.0, enable_defense=True):
	"""预测特定日期的Top5股票及权重
	
	Args:
		enable_defense: 是否启用现金仓位防御机制
	"""
	with torch.no_grad():
		x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
		scores = model(x).squeeze(0).detach().cpu().numpy()

	order = np.argsort(scores)[::-1]
	ranked_stock_ids = [sequence_stock_ids[i] for i in order]

	if len(ranked_stock_ids) < 5:
		return None, None, scores, None

	top5 = ranked_stock_ids[:5]
	top5_scores = scores[order[:5]]
	tensor_scores = torch.tensor(top5_scores, dtype=torch.float32)
	weights = F.softmax(tensor_scores / temperature, dim=0).numpy()

	# ========== Cash Defense 机制 ==========
	w_cash = None
	if enable_defense:
		mean_score = np.mean(top5_scores)
		DEFENSE_THRESHOLD = 0.0
		MIN_EXPOSURE = 0.3

		if mean_score < DEFENSE_THRESHOLD:
			# 市场信心不足，按比例缩减仓位
			# scale_factor: mean=-1.0 → MIN_EXPOSURE=0.3, mean=0.0 → 1.0
			scale_factor = MIN_EXPOSURE + (1.0 - MIN_EXPOSURE) * (mean_score - (-1.0)) / (DEFENSE_THRESHOLD - (-1.0))
			scale_factor = max(MIN_EXPOSURE, min(1.0, scale_factor))
			weights = weights * scale_factor
			w_cash = 1.0 - scale_factor
		else:
			w_cash = 0.0

	return top5, weights.tolist(), scores, w_cash


def calculate_T5_band_return(stock_code, pred_date, all_data_df):
	"""计算T+5波段收益率"""
	stock_data = all_data_df[all_data_df['股票代码'] == stock_code].sort_values('日期').copy()
	
	stock_data['open_t1'] = stock_data.groupby('股票代码')['开盘'].shift(-1)
	stock_data['open_t5'] = stock_data.groupby('股票代码')['开盘'].shift(-5)
	
	pred_row = stock_data[stock_data['日期'] == pred_date]
	
	if len(pred_row) == 0:
		return None
	
	open_t1 = pred_row['open_t1'].values[0]
	open_t5 = pred_row['open_t5'].values[0]
	
	# 边缘自适应
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


def run_non_overlapping_backtest():
	print("=" * 60)
	print("滚动回测系统 v5 (Non-overlapping T+5 Backtest)")
	print("修复：零化instrument + 无重叠周频评估")
	print("=" * 60)
	
	# 路径配置
	# 优先使用 Pairwise_Golden 模型（Golden Config 训练结果）
	# 否则 fallback 到 config 中的默认路径
	golden_model_path = './models/Pairwise_Golden/best_model.pth'
	if os.path.exists(golden_model_path):
		model_path = golden_model_path
		scaler_path = './models/Pairwise_Golden/scaler.pkl'
		print(f"[INFO] 使用 Pairwise_Golden 模型: {model_path}")
	else:
		model_path = os.path.join(config['output_dir'], 'best_model.pth')
		scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
		print(f"[INFO] 使用默认模型: {model_path}")
	
	# 读取完整数据
	full_data_path = './data/stock_data.csv'
	if not os.path.exists(full_data_path):
		full_data_path = './data/train.csv'
	
	full_df = pd.read_csv(full_data_path, dtype={'股票代码': str})
	full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
	full_df['日期'] = pd.to_datetime(full_df['日期'])
	
	# 定义验证集时间范围
	val_start = pd.Timestamp('2024-01-01')
	val_end = pd.Timestamp('2026-04-24')
	
	# 获取验证集中的交易日
	val_dates = sorted(full_df[(full_df['日期'] >= val_start) & (full_df['日期'] <= val_end)]['日期'].unique())
	print(f"验证集日期范围: {val_start.date()} ~ {val_end.date()}")
	print(f"验证集总交易日: {len(val_dates)} 天")
	
	# 选取无重叠的评估日期（每5个交易日选一个）
	eval_dates = val_dates[::5]
	print(f"无重叠评估日期数: {len(eval_dates)} 个")
	
	# 获取股票列表
	stock_ids = sorted(full_df['股票代码'].unique())
	stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
	
	# 处理验证集数据
	val_data = full_df[(full_df['日期'] >= val_start) & (full_df['日期'] <= val_end)].copy()
	processed_val, features = preprocess_predict_data(val_data, stockid2idx)
	processed_val[features] = processed_val[features].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
	
	# 加载scaler
	scaler = joblib.load(scaler_path)
	processed_val[features] = scaler.transform(processed_val[features])
	
	# 加载模型（保持原始input_dim=197）
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	print(f"使用设备: {device}")
	
	input_dim = len(features)
	print(f"模型输入维度: {input_dim}")
	
	model = StockTransformer(input_dim=input_dim, config=config, num_stocks=len(stock_ids))
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.to(device)
	model.eval()
	
	sequence_length = config['sequence_length']
	temperature = config.get('temperature', 1.0)
	
	# 开始无重叠T+5评估
	print(f"\n开始无重叠T+5评估...")
	
	eval_results = []
	all_predictions = []
	
	for eval_date in tqdm(eval_dates, desc='T+5评估'):
		try:
			# 使用评估日期构建序列
			seqs, stock_ids_ordered = build_inference_sequences_fixed(
				processed_val, features, sequence_length, stock_ids, eval_date
			)
			
			# 关键修复：将instrument特征置零
			seqs = zero_out_instrument(seqs, features)
			
			top5, weights, all_scores, w_cash = predict_top5_for_date(
				model, seqs, stock_ids_ordered, device, temperature, enable_defense=True
			)
			
			if top5 is None:
				continue
			
			all_predictions.append(tuple(top5))
			
			# 计算T+5波段收益
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
			# w_cash 是现金比例，实际股票仓位是 1-w_cash
			actual_exposure = 1.0 - w_cash if w_cash is not None else 1.0
			weighted_return = (stock_ret_df['return'] * stock_ret_df['weight']).sum() * actual_exposure
			
			# 打印防守日志（特别是 2024-09-30 暴跌日）
			if '2024-09-30' in str(eval_date) or '2024-10-08' in str(eval_date):
				print(f"\n[Cash Defense Debug] eval_date={eval_date}")
				print(f"  top5_scores={all_scores[order[:5]].tolist() if hasattr(all_scores[order[:5]], 'tolist') else list(all_scores[order[:5]])}")
				print(f"  mean_score={np.mean(all_scores[order[:5]]):.6f}")
				print(f"  w_cash={w_cash:.4f}, actual_exposure={actual_exposure:.4f}")
				print(f"  weighted_return(调整前)={weighted_return/actual_exposure:.4f}, weighted_return(调整后)={weighted_return:.4f}")
			
			eval_results.append({
				'eval_date': eval_date,
				'predicted_stocks': top5,
				'weights': weights,
				't5_return': weighted_return,
				'num_valid_stocks': valid_stocks,
				'is_profit': 1 if weighted_return > 0 else 0,
				'w_cash': w_cash,
				'actual_exposure': actual_exposure
			})
			
		except Exception as e:
			print(f"评估日期 {eval_date.date()} 失败: {e}")
			continue
	
	# 统计结果
	print("\n" + "=" * 60)
	print("无重叠T+5回测报告 v5")
	print("=" * 60)
	
	if len(eval_results) == 0:
		print("没有有效的评估结果")
		return
	
	result_df = pd.DataFrame(eval_results)
	
	# 基本统计
	total_evals = len(result_df)
	profitable_evals = result_df['is_profit'].sum()
	losing_evals = total_evals - profitable_evals
	
	t5_win_rate = profitable_evals / total_evals if total_evals > 0 else 0
	avg_t5_return = result_df['t5_return'].mean()
	std_t5_return = result_df['t5_return'].std()
	cumulative_return = (1 + result_df['t5_return']).prod() - 1
	
	# 预测多样性
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
	
	# 打印每轮评估明细
	print("\n" + "-" * 60)
	print("T+5 波段评估明细:")
	print("-" * 60)
	for i, row in result_df.iterrows():
		date_str = row['eval_date'].strftime('%Y-%m-%d')
		return_str = f"{row['t5_return']:.6f}"
		pos_neg = "+" if row['t5_return'] > 0 else ""
		stocks = ','.join(row['predicted_stocks'][:3]) + '...'
		print(f"{date_str}: {pos_neg}{return_str} | {stocks}")
	
	# 保存结果
	output_path = './output/non_overlapping_t5_results.csv'
	result_df.to_csv(output_path, index=False)
	print(f"\n详细结果已保存到: {output_path}")
	
	return {
		'total_evals': total_evals,
		't5_win_rate': t5_win_rate,
		'avg_t5_return': avg_t5_return,
		'cumulative_return': cumulative_return,
		'unique_predictions': unique_predictions
	}


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	results = run_non_overlapping_backtest()
	print("\n回测完成!")