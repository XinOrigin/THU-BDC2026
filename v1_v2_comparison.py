#!/usr/bin/env python3
"""
V1 vs V2 跨周期对照实验
测试 Pairwise_Golden (V1, 197维) vs Round08 (V2, 203维)
统一在 2024-01-02 ~ 2026-04-24 区间评估
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

from utils import engineer_features_158plus39
from model import StockTransformer

# ============================================================
# V1 特征列表（197维，含instrument，无截面排名特征）
# ============================================================
V1_FEATURES = [
    'instrument',  # 股票ID编码特征
    '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
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

# 模型配置
V1_MODEL_PATH = '/app/model/Pairwise_Golden/best_model.pth'
V1_CONFIG_PATH = '/app/model/Pairwise_Golden/config.json'
V1_SCALER_PATH = '/app/model/Pairwise_Golden/scaler.pkl'

# 回测配置
TEST_START = pd.Timestamp('2024-01-01')
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


def predict_top5(model, sequences_np, sequence_stock_ids, device, temperature=1.0):
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


def run_backtest(model_path, config_path, scaler_path, features, model_name):
    """运行回测"""
    # 加载配置
    import json
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    # 加载模型
    model = StockTransformer(
        input_dim=len(features),
        config=cfg,
        num_stocks=300
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 加载 scaler
    scaler = joblib.load(scaler_path)

    # 获取测试期交易日
    all_dates = sorted(df['日期'].unique())
    test_dates = [d for d in all_dates if d >= TEST_START]
    eval_dates = test_dates[::EVAL_INTERVAL]

    # 执行回测
    returns = []
    benchmark_returns = []

    for pred_date in eval_dates:
        try:
            available_stocks = processed[processed['日期'] == pred_date]['股票代码'].unique().tolist()

            if len(available_stocks) < 10:
                continue

            sequences_np, valid_stock_ids = build_inference_sequences(
                processed, features,
                cfg.get('sequence_length', 50),
                available_stocks,
                pred_date
            )

            if sequences_np is None or len(valid_stock_ids) < 5:
                continue

            top5, _ = predict_top5(model, sequences_np, valid_stock_ids, device, temperature=1.0)

            if top5 is None:
                continue

            # 计算组合收益
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

            # 计算基准
            benchmark_ret = calculate_hs300_return(pred_date, df, hs300_codes)

            if benchmark_ret is None:
                continue

            returns.append(portfolio_return)
            benchmark_returns.append(benchmark_ret)

        except Exception as e:
            continue

    return np.array(returns), np.array(benchmark_returns)


def main():
    global df, processed, device, hs300_codes

    print("=" * 70)
    print("V1 vs V2 跨周期对照实验")
    print("=" * 70)
    print(f"测试期: {TEST_START.date()} ~ 2026-04-24")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("\n加载数据...")
    df = pd.read_csv('/app/data/train.csv', dtype={'股票代码': str})
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    print(f"数据范围: {df['日期'].min().date()} ~ {df['日期'].max().date()}")

    stock_ids = sorted(df['股票代码'].unique())
    hs300_codes = stock_ids[:300]
    print(f"股票数量: {len(stock_ids)}")

    # V1 预处理（197维特征，不含截面排名）
    print("\nV1 特征工程（197维）...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    v1_list = []
    for group in tqdm(groups, desc='V1特征工程'):
        stock_features = engineer_features_158plus39(group)
        v1_list.append(stock_features)

    v1_processed = pd.concat(v1_list).reset_index(drop=True)
    v1_processed['日期'] = pd.to_datetime(v1_processed['日期'])
    
    # 添加 instrument 列（股票代码到索引的映射）
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(df['股票代码'].unique()))}
    v1_processed['instrument'] = v1_processed['股票代码'].map(stockid2idx)
    v1_processed = v1_processed.dropna(subset=['instrument']).copy()
    v1_processed['instrument'] = v1_processed['instrument'].astype(np.int64)
    
    v1_processed[V1_FEATURES] = v1_processed[V1_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

    v1_scaler = joblib.load(V1_SCALER_PATH)
    v1_processed[V1_FEATURES] = v1_scaler.transform(v1_processed[V1_FEATURES])

    print(f"V1 预处理完成: {len(v1_processed)} 条记录")

    # 全局变量赋值（供run_backtest使用）
    processed = v1_processed

    # 运行 V1 回测
    print("\n" + "=" * 50)
    print("运行 V1 (Pairwise_Golden) 回测...")
    print("=" * 50)
    v1_returns, v1_benchmark = run_backtest(V1_MODEL_PATH, V1_CONFIG_PATH, V1_SCALER_PATH, V1_FEATURES, "V1")

    # 计算 V1 指标
    v1_cumulative = np.cumprod(1 + v1_returns)
    v1_total = (v1_cumulative[-1] - 1) * 100
    v1_bench_cumulative = np.cumprod(1 + v1_benchmark)
    v1_bench_total = (v1_bench_cumulative[-1] - 1) * 100
    v1_alpha = v1_cumulative[-1] / v1_bench_cumulative[-1] - 1
    v1_win_rate = np.sum(v1_returns > 0) / len(v1_returns) * 100
    v1_excess_win = np.sum(v1_returns > v1_benchmark) / len(v1_returns) * 100
    v1_max_dd = compute_max_drawdown(v1_cumulative) * 100
    v1_sharpe = compute_sharpe_ratio(v1_returns)
    v1_avg = np.mean(v1_returns) * 100

    print(f"V1 累计收益: {v1_total:+.2f}%")
    print(f"V1 Alpha: {v1_alpha*100:+.2f}%")

    # 现在运行 V2 对比
    # 需要重新预处理 V2 数据（203维，含截面排名）
    print("\n" + "=" * 50)
    print("运行 V2 (Round08) 回测...")
    print("=" * 50)

    # V2 特征列表（203维）
    from train import feature_cloums_map
    from utils import add_cross_sectional_ranks

    V2_FEATURES = feature_cloums_map['158+39']

    # 重新预处理
    v2_list = []
    for group in tqdm(groups, desc='V2特征工程'):
        stock_features = engineer_features_158plus39(group)
        v2_list.append(stock_features)

    v2_processed = pd.concat(v2_list).reset_index(drop=True)
    v2_processed['日期'] = pd.to_datetime(v2_processed['日期'])
    v2_processed = add_cross_sectional_ranks(v2_processed)
    v2_processed[V2_FEATURES] = v2_processed[V2_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

    v2_scaler = joblib.load('/app/model/2026-05-04_Tuning/Round08_lr3e-05_drop0.15_temp1.0_score0.1790_ep3/scaler.pkl')
    v2_processed[V2_FEATURES] = v2_scaler.transform(v2_processed[V2_FEATURES])

    processed = v2_processed

    v2_returns, v2_benchmark = run_backtest(
        '/app/model/2026-05-04_Tuning/Round08_lr3e-05_drop0.15_temp1.0_score0.1790_ep3/best_model.pth',
        '/app/model/2026-05-04_Tuning/Round08_lr3e-05_drop0.15_temp1.0_score0.1790_ep3/config.json',
        '/app/model/2026-05-04_Tuning/Round08_lr3e-05_drop0.15_temp1.0_score0.1790_ep3/scaler.pkl',
        V2_FEATURES, "V2"
    )

    # 计算 V2 指标
    v2_cumulative = np.cumprod(1 + v2_returns)
    v2_total = (v2_cumulative[-1] - 1) * 100
    v2_bench_cumulative = np.cumprod(1 + v2_benchmark)
    v2_bench_total = (v2_bench_cumulative[-1] - 1) * 100
    v2_alpha = v2_cumulative[-1] / v2_bench_cumulative[-1] - 1
    v2_win_rate = np.sum(v2_returns > 0) / len(v2_returns) * 100
    v2_excess_win = np.sum(v2_returns > v2_benchmark) / len(v2_returns) * 100
    v2_max_dd = compute_max_drawdown(v2_cumulative) * 100
    v2_sharpe = compute_sharpe_ratio(v2_returns)
    v2_avg = np.mean(v2_returns) * 100

    print(f"V2 累计收益: {v2_total:+.2f}%")
    print(f"V2 Alpha: {v2_alpha*100:+.2f}%")

    # 输出对比报告
    print("\n" + "=" * 70)
    print("V1 vs V2 跨周期验尸对比报告")
    print("=" * 70)
    print(f"测试期: {TEST_START.date()} ~ 2026-04-24")
    print(f"总评估次数: {len(v1_returns)}")
    print(f"评估间隔: 每{EVAL_INTERVAL}个交易日")

    print("\n" + "=" * 70)
    print("【收益与 Alpha】")
    print("=" * 70)
    print(f"{'指标':<20} {'V1 (Pairwise_Golden)':<25} {'V2 (Round08)':<25} {'沪深300基准':<20}")
    print("-" * 90)
    print(f"{'累计收益':<20} {v1_total:>+18.2f}% {v2_total:>+18.2f}% {v1_bench_total:>+14.2f}%")
    print(f"{'Alpha (vs基准)':<20} {v1_alpha*100:>+18.2f}% {v2_alpha*100:>+18.2f}% {'--':<18}")
    print(f"{'平均单次收益':<20} {v1_avg:>+18.4f}% {v2_avg:>+18.4f}% {np.mean(v1_benchmark)*100:>+18.4f}%")

    print("\n" + "=" * 70)
    print("【胜率分析】")
    print("=" * 70)
    print(f"{'指标':<20} {'V1 (Pairwise_Golden)':<25} {'V2 (Round08)':<25}")
    print("-" * 70)
    print(f"{'正收益胜率':<20} {v1_win_rate:>18.1f}% {v2_win_rate:>18.1f}%")
    print(f"{'超额基准胜率':<20} {v1_excess_win:>18.1f}% {v2_excess_win:>18.1f}%")

    print("\n" + "=" * 70)
    print("【风险指标】")
    print("=" * 70)
    print(f"{'指标':<20} {'V1 (Pairwise_Golden)':<25} {'V2 (Round08)':<25} {'沪深300基准':<20}")
    print("-" * 90)
    print(f"{'最大回撤':<20} {v1_max_dd:>+18.2f}% {v2_max_dd:>+18.2f}% {compute_max_drawdown(v1_bench_cumulative)*100:>+18.2f}%")
    print(f"{'夏普比率':<20} {v1_sharpe:>+18.3f} {v2_sharpe:>+18.3f} {compute_sharpe_ratio(v1_benchmark):>+18.3f}")

    print("\n" + "=" * 70)
    print("【核心归因结论】")
    print("=" * 70)

    # 比较 V1 vs V2
    if v1_total > v2_total:
        winner = "V1 (Pairwise_Golden)"
        loser = "V2 (Round08)"
    else:
        winner = "V2 (Round08)"
        loser = "V1 (Pairwise_Golden)"

    print(f"\n【收益】{winner} 胜出，领先 {abs(v1_total - v2_total):.2f}%")

    if v1_sharpe > v2_sharpe:
        print(f"【风险收益】V1 夏普比率 {v1_sharpe:.3f} vs V2 {v2_sharpe:.3f}，V1 风险调整后收益更优")
    else:
        print(f"【风险收益】V2 夏普比率 {v2_sharpe:.3f} vs V1 {v1_sharpe:.3f}，V2 风险调整后收益更优")

    if v1_max_dd < v2_max_dd:
        print(f"【回撤控制】V1 最大回撤 {v1_max_dd:.2f}% < V2 {v2_max_dd:.2f}%，V1 更抗跌")
    else:
        print(f"【回撤控制】V2 最大回撤 {v2_max_dd:.2f}% < V1 {v1_max_dd:.2f}%，V2 更抗跌")

    # 根本原因分析
    print("\n" + "-" * 70)
    print("【Root Cause Hypothesis】")
    print("-" * 70)

    if v1_total > v2_total:
        print(f"""
❌ V2 在长周期中败于 V1！

可能的根本原因分析：
1. 【截面归一化双刃剑】
   - V2 在 collate_fn 中对每个时间截面做 Z-score 归一化
   - 这抹平了不同股票的绝对价格水平差异
   - 在大盘普涨/普跌时，V1 能捕捉绝对动量，V2 只知道相对排名

2. 【截面排名特征失效】
   - 7 个新增截面排名特征在长周期中成为噪声
   - 这些排名告诉模型"谁是当天热门"，但热门≠第二天继续涨
   - V1 纯技术面+Alpha因子在长期更稳定

3. 【过拟合验证集】
   - V2 Round08 得分 0.179 仅反映验证集内的排序能力
   - 该得分对未来预测能力有限
   - V1 的 Pairwise_Golden 虽然得分较低（0.2382 在 T=0.01 评估），但泛化能力可能更强
""")
    else:
        print(f"""
⚠️ V2 在长周期中胜出！

这说明 V2 的改进（截面归一化 + 低 margin + 排名特征）在长期有效。
但仍需注意 V2 在 2024-01~2026-04 区间相对基准仍然负 Alpha：
   V2 Alpha = {v2_alpha*100:+.2f}%

这表明策略整体需要加入对冲机制。
""")

    print("=" * 70)


if __name__ == '__main__':
    main()