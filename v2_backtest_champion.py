#!/usr/bin/env python3
"""
V2 王者模型全量回测脚本 - 非重叠 T+5 评估
测试 Round08 冠军模型 (lr=3e-05, dropout=0.15, margin=0.05, score=0.1790)

关键修复：
1. 使用与训练时一致的 '158+39' 特征列表（不包含 instrument，共 197 维）
2. 使用 walkforward_backtest_v5.py 经过验证的架构
3. 正确处理 scaler 维度匹配问题
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

from config import config
from model import StockTransformer
from utils import engineer_features_158plus39, add_cross_sectional_ranks

# ============================================================
# 配置
# ============================================================
CHAMPION_MODEL_DIR = '/app/model/2026-05-04_Tuning/Round08_lr3e-05_drop0.15_temp1.0_score0.1790_ep3'
MODEL_PATH = f'{CHAMPION_MODEL_DIR}/best_model.pth'
CONFIG_PATH = f'{CHAMPION_MODEL_DIR}/config.json'
SCALER_PATH = '/app/model/2026-05-04_Tuning/Round08_lr3e-05_drop0.15_temp1.0_score0.1790_ep3/scaler.pkl'
DATA_PATH = '/app/data/train.csv'

# 特征配置：'158+39' 不包含 instrument，共 203 维
feature_cloums_map = {
    '158+39': [
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
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread',
        # 截面排名特征
        '成交量_rank', '成交额_rank', '涨跌幅_rank', '换手率_rank',
        '涨跌幅_rank_change', '成交量_rank_change', '换手率_rank_change'
    ]
}


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
    """为特定日期构建推断序列"""
    sequences, sequence_stock_ids = [], []
    for stock_id in stock_ids:
        stock_history = data[
            (data['股票代码'] == stock_id) &
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)

        if len(stock_history) == sequence_length:
            sequences.append(stock_history[features].values.astype(np.float32))
            sequence_stock_ids.append(stock_id)

    if len(sequences) == 0:
        return None, None

    return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def predict_top5_for_date(model, sequences_np, sequence_stock_ids, device, temperature=1.0):
    """预测特定日期的Top5股票及权重"""
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
    """计算T+5波段收益率（用收盘价计算）"""
    stock_data = all_data_df[all_data_df['股票代码'] == stock_code].sort_values('日期').copy()

    # T+1 开盘价 = T+5 开盘前一日的收盘价（简化处理）
    stock_data['close_t1'] = stock_data.groupby('股票代码')['收盘'].shift(-1)
    stock_data['close_t5'] = stock_data.groupby('股票代码')['收盘'].shift(-5)

    pred_row = stock_data[stock_data['日期'] == pred_date]

    if len(pred_row) == 0:
        return None

    close_t1 = pred_row['close_t1'].values[0]
    close_t5 = pred_row['close_t5'].values[0]

    # 边缘自适应
    if pd.isna(close_t5):
        for shift_n in [4, 3, 2]:
            col_name = f'close_t{shift_n}'
            if col_name not in stock_data.columns:
                stock_data[col_name] = stock_data.groupby('股票代码')['收盘'].shift(-shift_n)
            close_t5 = pred_row[col_name].values[0]
            if not pd.isna(close_t5):
                break

    if pd.isna(close_t5) or close_t1 <= 0 or close_t5 <= 0:
        return None

    return (close_t5 - close_t1) / close_t1


def main():
    print("=" * 70)
    print("V2 王者模型全量回测 - 非重叠 T+5 评估")
    print("=" * 70)
    print(f"模型路径: {MODEL_PATH}")
    print(f"Scaler路径: {SCALER_PATH}")

    # 加载配置
    import json
    with open(CONFIG_PATH, 'r') as f:
        model_config = json.load(f)

    print(f"\n模型配置:")
    print(f"  - learning_rate: {model_config.get('learning_rate')}")
    print(f"  - dropout: {model_config.get('dropout')}")
    print(f"  - margin: {model_config.get('margin')}")
    print(f"  - d_model: {model_config.get('d_model')}")
    print(f"  - num_layers: {model_config.get('num_layers')}")
    print(f"  - feature_num: {model_config.get('feature_num')}")
    print(f"  - sequence_length: {model_config.get('sequence_length')}")

    # 加载数据
    print("\n加载数据...")
    df = pd.read_csv(DATA_PATH, dtype={'股票代码': str})
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    print(f"数据范围: {df['日期'].min().date()} ~ {df['日期'].max().date()}")
    print(f"股票数量: {df['股票代码'].nunique()}")

    # 定义特征列表（与训练一致）
    feature_num_key = '158+39'
    features = feature_cloums_map[feature_num_key]
    input_dim = len(features)
    print(f"\n特征维度: {input_dim}")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    print("\n加载模型...")
    num_stocks = 300  # 沪深300

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
    print(f"Scaler 加载完成 (n_features_in_: {scaler.n_features_in_})")

    # 定义回测周期
    train_end_date = pd.Timestamp('2026-02-13')
    test_start_date = pd.Timestamp('2026-02-24')  # 训练集后第一个交易日

    # 获取所有交易日
    all_dates = sorted(df['日期'].unique())
    test_dates = [d for d in all_dates if d >= test_start_date]
    print(f"\n测试期开始: {test_dates[0].date() if test_dates else 'N/A'}")
    print(f"测试期结束: {test_dates[-1].date() if test_dates else 'N/A'}")
    print(f"测试期天数: {len(test_dates)}")

    # 每5个交易日做一次预测（非重叠）
    eval_dates = test_dates[::5]
    print(f"预测次数: {len(eval_dates)}")

    # 获取股票列表
    stock_ids = sorted(df['股票代码'].unique())

    # 数据预处理：特征工程
    print("\n特征工程处理...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    processed_list = []
    for group in tqdm(groups, desc='特征工程'):
        stock_features = engineer_features_158plus39(group)
        processed_list.append(stock_features)

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['日期'] = pd.to_datetime(processed['日期'])
    print(f"特征工程完成: {len(processed)} 条记录")

    # 添加截面排名特征（关键！必须在 concat 后）
    print("添加截面排名特征...")
    processed = add_cross_sectional_ranks(processed)
    print(f"添加截面排名后: {len(processed)} 条记录")

    # 处理 NaN 和 Inf
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan)
    processed[features] = processed[features].fillna(0)

    # 应用 scaler
    print("应用 Scaler...")
    try:
        processed[features] = scaler.transform(processed[features])
        print("Scaler 应用成功")
    except Exception as e:
        print(f"Scaler 应用失败: {e}")
        return

    # 执行回测
    print("\n开始 T+5 非重叠回测...")
    results = []
    win_count = 0
    total_count = 0
    total_return = 0.0
    returns_list = []

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

            # 预测 Top5
            top5, weights = predict_top5_for_date(
                model, sequences_np, valid_stock_ids, device, temperature=1.0
            )

            if top5 is None:
                continue

            # 计算投资组合收益（等权）
            portfolio_return = 0.0
            valid_count = 0

            for stock_code in top5:
                ret = calculate_T5_return(stock_code, pred_date, df)
                if ret is not None:
                    portfolio_return += ret
                    valid_count += 1

            if valid_count > 0:
                portfolio_return /= valid_count
                returns_list.append(portfolio_return)
                total_return += portfolio_return
                total_count += 1
                if portfolio_return > 0:
                    win_count += 1

                results.append({
                    'pred_date': pred_date,
                    'top5': top5,
                    'weights': weights,
                    'portfolio_return': portfolio_return,
                    'win': portfolio_return > 0
                })

        except Exception as e:
            print(f"\n预测点 {pred_date} 出错: {e}")
            continue

    # 计算统计
    print("\n" + "=" * 70)
    print("V2 王者模型真实战力评估报告")
    print("=" * 70)

    if total_count > 0:
        avg_return = total_return / total_count
        win_rate = win_count / total_count * 100

        print(f"\n测试期: {test_dates[0].date()} ~ {test_dates[-1].date()}")
        print(f"总预测次数: {total_count}")
        print(f"胜率 (正收益占比): {win_rate:.2f}% ({win_count}/{total_count})")
        print(f"平均收益率: {avg_return*100:.4f}%")
        print(f"总绝对收益: {total_return*100:.4f}%")

        if returns_list:
            print(f"最高单次收益: {max(returns_list)*100:.4f}%")
            print(f"最低单次收益 (最大回撤): {min(returns_list)*100:.4f}%")

            # 分位数分析
            sorted_returns = sorted(returns_list)
            print(f"\n分位数分析:")
            q25_idx = int(len(sorted_returns) * 0.25)
            q50_idx = int(len(sorted_returns) * 0.5)
            q75_idx = int(len(sorted_returns) * 0.75)
            print(f"  25%: {sorted_returns[q25_idx]*100:.4f}%")
            print(f"  50%: {sorted_returns[q50_idx]*100:.4f}%")
            print(f"  75%: {sorted_returns[q75_idx]*100:.4f}%")

        # 保存结果
        result_df = pd.DataFrame(results)
        result_df.to_csv('/app/output/v2_champion_backtest_results.csv', index=False)
        print(f"\n结果已保存到: /app/output/v2_champion_backtest_results.csv")

        print("\n" + "=" * 70)
        if win_rate > 50:
            print(f"✅ 胜率 {win_rate:.2f}% 超越 50% 基准！")
        else:
            print(f"❌ 胜率 {win_rate:.2f}% 未突破 50% 基准")
        print("=" * 70)
    else:
        print("无有效回测结果")


if __name__ == '__main__':
    main()