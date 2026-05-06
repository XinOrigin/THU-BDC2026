#!/usr/bin/env python3
"""
V3 奇美拉模型长周期回测 - 《黎明战报》核心
- 测试期：2024-01-01 ~ 2026-04-24
- 引入沪深300基准对比
- 专业指标：Max Drawdown, Sharpe Ratio, Excess Win Rate
"""
import os
import sys
import joblib
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, '/app/code/src')

from train import feature_cloums_map
from utils import engineer_features_158plus39
from model import StockTransformer

# ============================================================
# 配置
# ============================================================
V3_MODEL_DIR = '/app/model/v3_158+39'
MODEL_PATH = f'{V3_MODEL_DIR}/Pairwise_Golden_V3_Best.pth'
CONFIG_PATH = f'{V3_MODEL_DIR}/config.json'
SCALER_PATH = f'{V3_MODEL_DIR}/scaler.pkl'
DATA_PATH = '/app/data/train.csv'

# 特征配置（V3: 197维，无截面排名特征）
features = feature_cloums_map['158+39']
input_dim = len(features)

# 回测配置
FULL_TEST_START = pd.Timestamp('2024-01-01')
EVAL_INTERVAL = 5


def build_inference_sequences(data, feature_cols, sequence_length, stock_ids, latest_date):
    """为特定日期构建推断序列"""
    sequences, sequence_stock_ids = [], []
    for stock_id in stock_ids:
        stock_history = data[
            (data['股票代码'] == stock_id) &
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)

        if len(stock_history) == sequence_length:
            sequences.append(stock_history[feature_cols].values.astype(np.float32))
            sequence_stock_ids.append(stock_id)

    if len(sequences) == 0:
        return None, None

    return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def predict_top5_for_date(model, sequences_np, sequence_stock_ids, device, temperature=1.0):
    """预测特定日期的Top5股票"""
    with torch.no_grad():
        x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
        scores = model(x).squeeze(0).detach().cpu().numpy()

    order = np.argsort(scores)[::-1]
    ranked_stock_ids = [sequence_stock_ids[i] for i in order]

    if len(ranked_stock_ids) < 5:
        return None, None

    top5 = ranked_stock_ids[:5]
    top5_scores = scores[order[:5]]
    tensor_scores = torch.tensor(top5_scores, dtype=torch.float32)
    weights = F.softmax(tensor_scores / temperature, dim=0).numpy()

    return top5, weights.tolist()


def calculate_T5_return(stock_code, pred_date, all_data_df):
    """计算T+5收益"""
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


def calculate_hs300_return(pred_date, all_data_df, hs300_codes):
    """计算同日沪深300指数收益率"""
    returns = []
    for code in hs300_codes:
        stock_data = all_data_df[all_data_df['股票代码'] == code].sort_values('日期').copy()
        stock_data['open_t1'] = stock_data.groupby('股票代码')['开盘'].shift(-1)
        stock_data['open_t5'] = stock_data.groupby('股票代码')['开盘'].shift(-5)

        pred_row = stock_data[stock_data['日期'] == pred_date]
        if len(pred_row) == 0:
            continue

        open_t1 = pred_row['open_t1'].values[0]
        open_t5 = pred_row['open_t5'].values[0]

        if pd.isna(open_t5) or open_t1 <= 0 or open_t5 <= 0:
            continue

        ret = (open_t5 - open_t1) / open_t1
        returns.append(ret)

    if len(returns) == 0:
        return None

    return np.mean(returns)


def compute_max_drawdown(cumulative_returns):
    """计算最大回撤"""
    peak = cumulative_returns[0]
    max_dd = 0.0
    for ret in cumulative_returns:
        if ret > peak:
            peak = ret
        dd = peak - ret
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_sharpe_ratio(returns, risk_free_rate=0.03):
    """计算年化夏普比率"""
    if len(returns) < 2:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return 0.0

    annual_return = mean_return * 252
    annual_std = std_return * np.sqrt(252)

    sharpe = (annual_return - risk_free_rate) / annual_std
    return sharpe


def main():
    print("=" * 70)
    print("V3 奇美拉模型长周期回测")
    print("=" * 70)
    print(f"测试期: {FULL_TEST_START.date()} ~ 2026-04-24")

    # 加载配置
    with open(CONFIG_PATH, 'r') as f:
        model_config = json.load(f)

    print(f"\n模型配置:")
    print(f"  - margin={model_config.get('margin')}, lr={model_config.get('learning_rate')}")
    print(f"  - dropout={model_config.get('dropout')}, batch_size={model_config.get('batch_size')}")
    print(f"  - feature_dim={input_dim} (197维，无截面排名特征)")

    # 加载数据
    print("\n加载数据...")
    df = pd.read_csv(DATA_PATH, dtype={'股票代码': str})
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    print(f"数据范围: {df['日期'].min().date()} ~ {df['日期'].max().date()}")

    # HS300成分股
    stock_ids = sorted(df['股票代码'].unique())
    hs300_codes = stock_ids[:300]
    print(f"股票数量: {len(stock_ids)}")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    print("\n加载V3模型...")
    num_stocks = 300

    model = StockTransformer(
        input_dim=input_dim,
        config=model_config,
        num_stocks=num_stocks
    ).to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("模型加载完成")

    # 加载 scaler
    print("加载 Scaler...")
    scaler = joblib.load(SCALER_PATH)
    print(f"Scaler n_features_in_: {scaler.n_features_in_}")

    # 获取测试期交易日
    all_dates = sorted(df['日期'].unique())
    test_dates = [d for d in all_dates if d >= FULL_TEST_START]
    print(f"\n测试期总交易日: {len(test_dates)}")

    eval_dates = test_dates[::EVAL_INTERVAL]
    print(f"评估次数: {len(eval_dates)}")

    # 数据预处理（V3: 无add_cross_sectional_ranks）
    print("\n特征工程（V3: 纯197维原始特征）...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    processed_list = []
    for group in tqdm(groups, desc='特征工程'):
        stock_features = engineer_features_158plus39(group)
        processed_list.append(stock_features)

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['日期'] = pd.to_datetime(processed['日期'])

    # V3: 不添加截面排名特征
    print(f"预处理完成: {len(processed)} 条记录（无截面排名特征）")

    # 处理 NaN/Inf
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 应用 scaler
    print("应用 Scaler...")
    processed[features] = scaler.transform(processed[features])

    # 执行回测
    print("\n开始V3长周期回测...")
    results = []
    v3_returns = []
    benchmark_returns = []

    for pred_date in tqdm(eval_dates, desc='回测进度'):
        try:
            available_stocks = processed[processed['日期'] == pred_date]['股票代码'].unique().tolist()

            if len(available_stocks) < 10:
                continue

            sequences_np, valid_stock_ids = build_inference_sequences(
                processed, features,
                model_config.get('sequence_length', 50),
                available_stocks,
                pred_date
            )

            if sequences_np is None or len(valid_stock_ids) < 5:
                continue

            top5, weights = predict_top5_for_date(
                model, sequences_np, valid_stock_ids, device, temperature=1.0
            )

            if top5 is None:
                continue

            portfolio_return = 0.0
            valid_count = 0

            for stock_code in top5:
                ret = calculate_T5_return(stock_code, pred_date, df)
                if ret is not None:
                    portfolio_return += ret
                    valid_count += 1

            if valid_count == 0:
                continue

            portfolio_return /= valid_count

            benchmark_ret = calculate_hs300_return(pred_date, df, hs300_codes)

            if benchmark_ret is None:
                continue

            v3_returns.append(portfolio_return)
            benchmark_returns.append(benchmark_ret)

            results.append({
                'pred_date': pred_date,
                'v3_return': portfolio_return,
                'benchmark_return': benchmark_ret,
                'excess_return': portfolio_return - benchmark_ret
            })

        except Exception as e:
            continue

    # ============================================================
    # 计算最终指标
    # ============================================================
    print("\n" + "=" * 70)
    print("V3 长周期回测结果")
    print("=" * 70)

    if len(v3_returns) == 0:
        print("无有效回测结果！")
        return

    results_df = pd.DataFrame(results)
    results_df.to_csv('/app/output/v3_long_term_results.csv', index=False)

    # 累计收益
    cumulative_v3 = [(1 + r) for r in v3_returns]
    cumulative_bench = [(1 + r) for r in benchmark_returns]

    for i in range(1, len(cumulative_v3)):
        cumulative_v3[i] *= cumulative_v3[i-1]
        cumulative_bench[i] *= cumulative_bench[i-1]

    cumulative_v3 = [x - 1 for x in cumulative_v3]
    cumulative_bench = [x - 1 for x in cumulative_bench]

    total_return_v3 = cumulative_v3[-1] * 100
    total_return_bench = cumulative_bench[-1] * 100

    # Alpha
    alpha = total_return_v3 - total_return_bench

    # 胜率
    win_count = sum(1 for i in range(len(v3_returns)) if v3_returns[i] > benchmark_returns[i])
    win_rate = win_count / len(v3_returns) * 100

    # 最大回撤
    max_dd = compute_max_drawdown(cumulative_v3) * 100

    # 夏普比率
    sharpe = compute_sharpe_ratio(v3_returns)

    # 超额收益
    excess_return = total_return_v3 - total_return_bench

    print(f"\n评估次数: {len(v3_returns)}")
    print(f"\n【收益指标】")
    print(f"  V3累计收益: {total_return_v3:.2f}%")
    print(f"  沪深300基准: {total_return_bench:.2f}%")
    print(f"  Alpha超额: {alpha:.2f}%")

    print(f"\n【胜率指标】")
    print(f"  超额胜率: {win_rate:.2f}% ({win_count}/{len(v3_returns)})")

    print(f"\n【风险指标】")
    print(f"  最大回撤: {max_dd:.2f}%")
    print(f"  夏普比率: {sharpe:.4f}")

    # 保存结果
    report = {
        'total_return_v3': total_return_v3,
        'total_return_bench': total_return_bench,
        'alpha': alpha,
        'win_rate': win_rate,
        'win_count': win_count,
        'total_evaluations': len(v3_returns),
        'max_drawdown': max_dd,
        'sharpe_ratio': sharpe,
        'excess_return': excess_return
    }

    with open('/app/output/v3_metrics.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n结果已保存到 /app/output/v3_long_term_results.csv")
    print(f"指标已保存到 /app/output/v3_metrics.json")

    return report


if __name__ == '__main__':
    main()
