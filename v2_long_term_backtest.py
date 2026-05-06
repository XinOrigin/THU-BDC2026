#!/usr/bin/env python3
"""
V2 王者模型长周期极限压力测试
- 测试期：2024-01-01 ~ 2026-04-24（覆盖完整涨跌震荡周期）
- 引入沪深300基准对比
- 专业指标：Max Drawdown, Sharpe Ratio, Excess Win Rate
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, '/app/code/src')

from train import feature_cloums_map
from utils import engineer_features_158plus39, add_cross_sectional_ranks
from model import StockTransformer

# ============================================================
# 配置
# ============================================================
CHAMPION_MODEL_DIR = '/app/model/2026-05-04_Tuning/Round08_lr3e-05_drop0.15_temp1.0_score0.1790_ep3'
MODEL_PATH = f'{CHAMPION_MODEL_DIR}/best_model.pth'
CONFIG_PATH = f'{CHAMPION_MODEL_DIR}/config.json'
SCALER_PATH = f'{CHAMPION_MODEL_DIR}/scaler.pkl'
DATA_PATH = '/app/data/train.csv'

# 特征配置
features = feature_cloums_map['158+39']
input_dim = len(features)

# 回测配置
FULL_TEST_START = pd.Timestamp('2024-01-01')  # 扩大测试期到1年前
EVAL_INTERVAL = 5  # 每5个交易日评估一次


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
    """计算T+5收益（从T+1开盘买入，T+5开盘卖出）"""
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


def calculate_hs300_return(pred_date, all_data_df, hs300_codes):
    """计算同日沪深300指数收益率（T+1开盘到T+5开盘）"""
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
    """计算年化夏普比率（假设252交易日）"""
    if len(returns) < 2:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return 0.0

    # 年化
    annual_return = mean_return * 252
    annual_std = std_return * np.sqrt(252)

    sharpe = (annual_return - risk_free_rate) / annual_std
    return sharpe


def main():
    print("=" * 70)
    print("V2 王者模型长周期极限压力测试")
    print("=" * 70)
    print(f"测试期: {FULL_TEST_START.date()} ~ 2026-04-24")

    # 加载配置
    import json
    with open(CONFIG_PATH, 'r') as f:
        model_config = json.load(f)

    print(f"\n模型配置:")
    print(f"  - lr={model_config.get('learning_rate')}, dropout={model_config.get('dropout')}")
    print(f"  - d_model={model_config.get('d_model')}, num_layers={model_config.get('num_layers')}")

    # 加载数据
    print("\n加载数据...")
    df = pd.read_csv(DATA_PATH, dtype={'股票代码': str})
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    print(f"数据范围: {df['日期'].min().date()} ~ {df['日期'].max().date()}")

    # 获取HS300成分股（用于基准计算）
    stock_ids = sorted(df['股票代码'].unique())
    hs300_codes = stock_ids[:300]  # 假设前300只是HS300成分
    print(f"股票数量: {len(stock_ids)}")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    print("\n加载模型...")
    num_stocks = 300

    model = StockTransformer(
        input_dim=input_dim,
        config=model_config,
        num_stocks=num_stocks
    ).to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device)
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

    # 评估日期（每5个交易日一次）
    eval_dates = test_dates[::EVAL_INTERVAL]
    print(f"评估次数: {len(eval_dates)}")

    # 数据预处理
    print("\n特征工程...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    processed_list = []
    for group in tqdm(groups, desc='特征工程'):
        stock_features = engineer_features_158plus39(group)
        processed_list.append(stock_features)

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['日期'] = pd.to_datetime(processed['日期'])

    # 添加截面排名特征
    print("添加截面排名特征...")
    processed = add_cross_sectional_ranks(processed)
    print(f"预处理完成: {len(processed)} 条记录")

    # 处理 NaN/Inf
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 应用 scaler
    print("应用 Scaler...")
    processed[features] = scaler.transform(processed[features])

    # 执行回测
    print("\n开始长周期回测...")
    results = []
    v2_returns = []  # V2策略收益
    benchmark_returns = []  # 基准收益

    for pred_date in tqdm(eval_dates, desc='回测进度'):
        try:
            # 获取预测日可用的股票
            available_stocks = processed[processed['日期'] == pred_date]['股票代码'].unique().tolist()

            if len(available_stocks) < 10:
                continue

            # 构建序列
            sequences_np, valid_stock_ids = build_inference_sequences(
                processed, features,
                model_config.get('sequence_length', 50),
                available_stocks,
                pred_date
            )

            if sequences_np is None or len(valid_stock_ids) < 5:
                continue

            # V2预测
            top5, weights = predict_top5_for_date(
                model, sequences_np, valid_stock_ids, device, temperature=1.0
            )

            if top5 is None:
                continue

            # 计算V2投资组合收益
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

            # 计算基准（沪深300等权）收益
            benchmark_ret = calculate_hs300_return(pred_date, df, hs300_codes)

            if benchmark_ret is None:
                continue

            v2_returns.append(portfolio_return)
            benchmark_returns.append(benchmark_ret)

            results.append({
                'pred_date': pred_date,
                'v2_return': portfolio_return,
                'benchmark_return': benchmark_ret,
                'excess_return': portfolio_return - benchmark_ret
            })

        except Exception as e:
            continue

    # 计算专业指标
    print("\n" + "=" * 70)
    print("V2 王者模型跨周期极限体检报告")
    print("=" * 70)

    if len(v2_returns) == 0:
        print("无有效回测结果")
        return

    # 转换为numpy数组
    v2_returns = np.array(v2_returns)
    benchmark_returns = np.array(benchmark_returns)

    # 累计收益曲线
    v2_cumulative = np.cumprod(1 + v2_returns)
    benchmark_cumulative = np.cumprod(1 + benchmark_returns)

    # 基本统计
    total_v2_return = (v2_cumulative[-1] - 1) * 100
    total_benchmark_return = (benchmark_cumulative[-1] - 1) * 100

    win_count = np.sum(v2_returns > 0)
    exceed_benchmark_count = np.sum(v2_returns > benchmark_returns)
    total_count = len(v2_returns)

    win_rate = win_count / total_count * 100
    excess_win_rate = exceed_benchmark_count / total_count * 100

    avg_return = np.mean(v2_returns) * 100
    avg_benchmark = np.mean(benchmark_returns) * 100

    # 最大回撤
    v2_max_drawdown = compute_max_drawdown(v2_cumulative) * 100
    benchmark_max_drawdown = compute_max_drawdown(benchmark_cumulative) * 100

    # 夏普比率
    v2_sharpe = compute_sharpe_ratio(v2_returns)
    benchmark_sharpe = compute_sharpe_ratio(benchmark_returns)

    # 超额收益
    excess_returns = v2_returns - benchmark_returns
    total_excess_return = (v2_cumulative[-1] / benchmark_cumulative[-1] - 1) * 100

    print(f"\n【测试概览】")
    print(f"测试期: {test_dates[0].date()} ~ {test_dates[-1].date()}")
    print(f"总评估次数: {total_count}")
    print(f"评估间隔: 每{EVAL_INTERVAL}个交易日")

    print(f"\n【收益对比】")
    print(f"V2策略累计收益:    {total_v2_return:+.2f}%")
    print(f"沪深300基准收益:    {total_benchmark_return:+.2f}%")
    print(f"超额收益 (Alpha):   {total_excess_return:+.2f}%")

    print(f"\n【胜率分析】")
    print(f"V2正收益胜率:      {win_rate:.1f}% ({win_count}/{total_count})")
    print(f"超额基准胜率:       {excess_win_rate:.1f}% ({exceed_benchmark_count}/{total_count})")

    print(f"\n【风险指标】")
    print(f"V2最大回撤:        {v2_max_drawdown:.2f}%")
    print(f"沪深300最大回撤:    {benchmark_max_drawdown:.2f}%")
    print(f"V2夏普比率:         {v2_sharpe:.3f}")
    print(f"沪深300夏普比率:     {benchmark_sharpe:.3f}")

    print(f"\n【收益分布】")
    print(f"V2平均收益:        {avg_return:+.4f}%")
    print(f"基准平均收益:       {avg_benchmark:+.4f}%")
    print(f"V2最高单次:        {np.max(v2_returns)*100:+.2f}%")
    print(f"V2最低单次:        {np.min(v2_returns)*100:+.2f}%")

    # 分位数
    sorted_v2 = np.sort(v2_returns)
    print(f"V2收益分位数:")
    print(f"  25%: {sorted_v2[int(len(sorted_v2)*0.25)]*100:+.2f}%")
    print(f"  50%: {sorted_v2[int(len(sorted_v2)*0.5)]*100:+.2f}%")
    print(f"  75%: {sorted_v2[int(len(sorted_v2)*0.75)]*100:+.2f}%")

    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv('/app/output/v2_long_term_results.csv', index=False)
    print(f"\n详细结果已保存到: /app/output/v2_long_term_results.csv")

    # 总结
    print("\n" + "=" * 70)
    print("【极限体检结论】")

    passed = []
    failed = []

    # Alpha检验
    if total_excess_return > 0:
        passed.append(f"✅ Alpha为正: {total_excess_return:+.2f}%")
    else:
        failed.append(f"❌ Alpha为负: {total_excess_return:+.2f}%")

    # 超额胜率检验
    if excess_win_rate > 50:
        passed.append(f"✅ 超额胜率: {excess_win_rate:.1f}%")
    else:
        failed.append(f"❌ 超额胜率: {excess_win_rate:.1f}%")

    # 最大回撤检验
    if v2_max_drawdown < benchmark_max_drawdown:
        passed.append(f"✅ 回撤控制: V2={v2_max_drawdown:.1f}% vs 基准={benchmark_max_drawdown:.1f}%")
    else:
        failed.append(f"⚠️ 回撤超标: V2={v2_max_drawdown:.1f}% vs 基准={benchmark_max_drawdown:.1f}%")

    # 夏普比率检验
    if v2_sharpe > benchmark_sharpe:
        passed.append(f"✅ 风险收益比: V2={v2_sharpe:.3f} vs 基准={benchmark_sharpe:.3f}")
    else:
        failed.append(f"⚠️ 风险收益比: V2={v2_sharpe:.3f} vs 基准={benchmark_sharpe:.3f}")

    for msg in passed:
        print(msg)
    for msg in failed:
        print(msg)

    print("=" * 70)


if __name__ == '__main__':
    main()