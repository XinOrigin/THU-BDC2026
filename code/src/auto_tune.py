#!/usr/bin/env python3
"""
自动化模型调优脚本
自动执行多轮训练和参数搜索，结果保存在以当天日期命名的专属文件夹中

安全特性：
- 数据只加载一次，在循环外部
- 每轮训练后彻底清理GPU显存
- 状态完全隔离，每轮从零开始
"""

import os
import sys
import json
import random
import datetime
import gc
import shutil
import numpy as np
import pandas as pd
import torch
import joblib
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

# 添加 src 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config as base_config, sequence_length, feature_num
from train import (
    set_seed, split_train_val_by_last_month, preprocess_data, preprocess_val_data,
    StockTransformer, RankingDataset, collate_fn,
    train_ranking_model, evaluate_ranking_model,
    WeightedRankingLoss, PairwiseRankingLoss, calculate_ranking_metrics, create_ranking_dataset_vectorized
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ============== V2 专用基础配置 ==============
BASE_CONFIG = {
    'sequence_length': sequence_length,
    'feature_num': feature_num,
    'd_model': 256,  # V2: golden config
    'nhead': 16,
    'num_layers': 3,  # V2: golden config
    'dim_feedforward': 512,
    'batch_size': 8,  # V2: reduced to 8 to prevent OOM
    'num_epochs': 25,
    'dropout': 0.15,  # V2: reduced from 0.3
    'max_grad_norm': 5.0,
    'pairwise_weight': 1,
    'base_weight': 1.0,
    'top5_weight': 2.0,
    'early_stopping_patience': 8,  # V2: reduced from 10
    'warmup_epochs': 2,
    'data_path': '/app/data',
    'accumulation_steps': 4,  # V2: gradient accumulation for effective batch size of 32
}

# ============== V2 专用参数搜索空间 ==============
PARAM_GRID = {
    'learning_rate': [3e-5, 5e-5, 8e-5],  # V2: LayerNorm stabilizes, use slightly lower LR
    'dropout': [0.1, 0.15, 0.2],  # V2: reduced from V1 due to LayerNorm
    'margin': [0.01, 0.05, 0.1],  # V2: tighter margins for finer ranking
    'd_model': [256, 384],  # V2: 256 golden, 384 optional
    'num_layers': [2, 3],  # V2: 3 golden, 2 for faster training
    'warmup_epochs': [2, 3],  # V2: slightly more warmup
}

# ============== 实验设置 ==============
NUM_ROUNDS = 12  # V2: 10-15 rounds for overnight tuning
TOP_K_BEST = 3   # 保存最好的K个模型


def get_date_folder():
    """获取当天的专属实验文件夹路径"""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    base_model_dir = '/app/model'
    exp_dir = os.path.join(base_model_dir, f'{today}_Tuning')
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir, today


def sample_params():
    """随机采样参数组合"""
    params = {}
    for key, values in PARAM_GRID.items():
        params[key] = random.choice(values)
    return params


def build_config(round_idx, params):
    """构建当前轮次的配置"""
    config = deepcopy(BASE_CONFIG)
    config.update(params)
    config['output_dir'] = f'./models/temp_round_{round_idx}'
    return config


def load_data_once(data_path, sequence_length):
    """
    【关键优化】数据只加载一次，在循环外部
    返回预处理的训练/验证数据和特征列表
    """
    print("="*60)
    print("正在加载并预处理数据（仅执行一次）...")
    print("="*60)
    
    data_file = os.path.join(data_path, 'train.csv')
    full_df = pd.read_csv(data_file)
    train_df, val_df, val_start = split_train_val_by_last_month(full_df, sequence_length)
    
    # 获取所有股票ID，建立映射
    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)
    
    # 特征工程与预处理
    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)
    
    # 标准化
    scaler = StandardScaler()
    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    
    # 创建排序数据集
    train_sequences, train_targets, train_relevance, train_stock_indices = create_ranking_dataset_vectorized(
        train_data, features, sequence_length, ranking_data_path=None
    )
    val_sequences, val_targets, val_relevance, val_stock_indices = create_ranking_dataset_vectorized(
        val_data, features, sequence_length, ranking_data_path=None
    )
    
    train_dataset = RankingDataset(train_sequences, train_targets, train_relevance, train_stock_indices)
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BASE_CONFIG['batch_size'], shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BASE_CONFIG['batch_size'], shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )
    
    print(f"数据加载完成: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本")
    print(f"特征数量: {len(features)}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'features': features,
        'num_stocks': num_stocks,
        'scaler': scaler,
        'train_data': train_data,
        'val_data': val_data,
    }


def run_training_round(round_idx, config, device, exp_dir, shared_data):
    """
    执行单轮训练（使用预加载的数据）
    返回: (best_score, best_epoch, final_loss, early_stopped)
    """
    print(f"\n{'='*60}")
    print(f"Round {round_idx} 开始训练")
    print(f"参数: lr={config['learning_rate']:.1e}, drop={config['dropout']}, temp={config.get('temperature', 1.0)}")
    print(f"{'='*60}")
    
    set_seed(config.get('seed', 42 + round_idx))
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log'))
    
    # 【关键】使用预加载的数据
    train_loader = shared_data['train_loader']
    val_loader = shared_data['val_loader']
    features = shared_data['features']
    num_stocks = shared_data['num_stocks']
    
    # 【关键】模型从零开始初始化
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.to(device)
    
    # 优化器从零开始
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    
    # 损失函数 - 使用 PairwiseRankingLoss (方案C)，margin 从参数采样
    criterion = PairwiseRankingLoss(margin=config.get('margin', 1.0), k=5)
    
    # 学习率调度
    warmup_epochs = config.get('warmup_epochs', 2)
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        decay_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.2,
            total_iters=max(1, config['num_epochs'] - warmup_epochs)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.2, total_iters=config['num_epochs']
        )
    
    # 训练循环
    best_score = -float('inf')
    best_epoch = -1
    patience = config.get('early_stopping_patience', 10)
    counter = 0
    early_stopped = False
    final_loss = 0.0
    
    for epoch in range(config['num_epochs']):
        print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")
        
        # 训练
        train_loss, train_metrics = train_ranking_model(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            accumulation_steps=config.get('accumulation_steps', 4)
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # 验证
        eval_loss, eval_metrics = evaluate_ranking_model(
            model, val_loader, criterion, device, writer, epoch
        )
        print(f"Eval Loss: {eval_loss:.4f}")
        print(f"Eval final_score: {eval_metrics.get('final_score', 0.0):.4f}")
        
        # 学习率调度
        scheduler.step()
        
        # 保存最佳模型
        current_final_score = eval_metrics.get('final_score', 0.0)
        if current_final_score > best_score:
            best_score = current_final_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"保存最佳模型 - final score: {best_score:.4f}")
            counter = 0
        else:
            counter += 1
            print(f"验证集性能未提升 ({counter}/{patience})")
            if counter >= patience:
                print(f"\n早停触发：验证集性能连续 {patience} 个epoch未提升，停止训练")
                print(f"最佳模型来自 epoch {best_epoch}，final score: {best_score:.4f}")
                early_stopped = True
                final_loss = eval_loss
                break
        
        final_loss = eval_loss
    
    # 加载最佳模型权重
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    
    writer.close()
    
    # 【关键】显存清理
    del model
    del optimizer
    del scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return best_score, best_epoch, final_loss, early_stopped


def save_model_with_name(exp_dir, round_idx, best_score, best_epoch, config, rank):
    """将模型保存为规范文件夹格式
    
    每个模型保存在独立文件夹中，包含：
    - config.json: 模型配置
    - best_model.pth: 模型权重
    - result.csv: 预测结果（如果有）
    - final_score.txt: 最终评分
    """
    model_path = f'./models/temp_round_{round_idx}/best_model.pth'
    if not os.path.exists(model_path):
        return None
    
    lr = config.get('learning_rate', 4e-5)
    drop = config.get('dropout', 0.2)
    temp = config.get('temperature', 1.0)
    
    # 创建规范化的文件夹名称
    folder_name = f"Round{round_idx:02d}_lr{lr}_drop{drop}_temp{temp}_score{best_score:.4f}_ep{best_epoch}"
    model_folder = os.path.join(exp_dir, folder_name)
    os.makedirs(model_folder, exist_ok=True)
    
    # 复制配置文件
    config_path = f'./models/temp_round_{round_idx}/config.json'
    if os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(model_folder, 'config.json'))
    
    # 复制模型权重
    shutil.copy(model_path, os.path.join(model_folder, 'best_model.pth'))
    
    # 复制 scaler 如果存在
    scaler_path = f'./models/temp_round_{round_idx}/scaler.pkl'
    if os.path.exists(scaler_path):
        shutil.copy(scaler_path, os.path.join(model_folder, 'scaler.pkl'))
    
    # 复制 result.csv 如果存在
    result_path = f'./models/temp_round_{round_idx}/result.csv'
    if os.path.exists(result_path):
        shutil.copy(result_path, os.path.join(model_folder, 'result.csv'))
    
    # 保存最终评分
    score_path = os.path.join(model_folder, 'final_score.txt')
    with open(score_path, 'w', encoding='utf-8') as f:
        f.write(f"final_score: {best_score:.6f}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"rank: {rank}\n")
        f.write(f"learning_rate: {lr}\n")
        f.write(f"dropout: {drop}\n")
        f.write(f"temperature: {temp}\n")
    
    return folder_name


def update_tuning_log(exp_dir, round_idx, params, best_score, best_epoch, final_loss, early_stopped, prev_score, trend):
    """更新实验日志"""
    log_path = os.path.join(exp_dir, 'tuning_log.md')
    
    if round_idx == 1:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"# 自动调优实验日志\n")
            f.write(f"实验日期: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"## Round {round_idx}\n\n")
        f.write(f"**参数组合**: {json.dumps(params, indent=2, ensure_ascii=False)}\n\n")
        f.write(f"**训练结果**:\n")
        f.write(f"- 最高得分 (final_score): **{best_score:.6f}**\n")
        f.write(f"- 最佳 Epoch: {best_epoch}\n")
        f.write(f"- 最终 Loss: {final_loss:.6f}\n")
        f.write(f"- 早停触发: {'是' if early_stopped else '否'}\n\n")
        
        if prev_score is not None:
            delta = best_score - prev_score
            f.write(f"**对比上一轮**: {trend} (delta: {delta:+.6f})\n\n")
        else:
            f.write(f"**对比上一轮**: 基准轮\n\n")
        
        f.write("---\n\n")


def cleanup_temp_dirs(round_idx):
    """清理临时目录"""
    temp_dir = f'./models/temp_round_{round_idx}'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def main():
    """主函数"""
    print("="*60)
    print("自动化模型调优 Pipeline")
    print("="*60)
    
    # 创建立专属实验文件夹
    exp_dir, today = get_date_folder()
    print(f"\n实验保存路径: {exp_dir}")
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 【关键】数据只加载一次，在循环外部
    shared_data = load_data_once(BASE_CONFIG['data_path'], BASE_CONFIG['sequence_length'])
    
    # 记录历史最佳
    all_results = []
    prev_score = None
    best_overall_score = -float('inf')
    
    # 多轮训练循环
    for round_idx in range(1, NUM_ROUNDS + 1):
        print(f"\n{'#'*60}")
        print(f"# Round {round_idx} / {NUM_ROUNDS}")
        print(f"{'#'*60}")
        
        # 采样参数
        params = sample_params()
        print(f"采样参数: lr={params['learning_rate']:.1e}, drop={params['dropout']}, temp={params.get('temperature', 1.0)}")
        
        # 构建配置
        config = build_config(round_idx, params)
        
        # 执行训练（使用预加载的数据）
        # 【关键】添加异常处理，防止单轮崩溃导致整个程序终止
        try:
            best_score, best_epoch, final_loss, early_stopped = run_training_round(
                round_idx, config, device, exp_dir, shared_data
            )
        except Exception as e:
            print(f"\n❌ Round {round_idx} 训练失败: {e}")
            print("跳过此轮，继续下一轮...")
            # 记录失败
            all_results.append({
                'round': round_idx,
                'params': params,
                'best_score': -999.0,  # 失败标记
                'best_epoch': -1,
                'final_loss': -1.0,
                'early_stopped': False,
                'model_name': None,
                'error': str(e)
            })
            # 清理残留
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cleanup_temp_dirs(round_idx)
            continue
        
        # 判断趋势
        if prev_score is None:
            trend = "基准"
        elif best_score > prev_score:
            trend = "⬆️ 变好"
        elif best_score < prev_score:
            trend = "⬇️ 变差"
        else:
            trend = "➡️ 持平"
        
        # 更新日志
        update_tuning_log(exp_dir, round_idx, params, best_score, best_epoch, 
                         final_loss, early_stopped, prev_score, trend)
        
        # 保存模型（使用规范命名）
        rank = len(all_results) + 1
        model_name = save_model_with_name(exp_dir, round_idx, best_score, best_epoch, config, rank)
        print(f"模型已保存: {model_name}")
        
        # 清理临时目录
        cleanup_temp_dirs(round_idx)
        
        # 【额外安全检查】清理本轮残留
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 记录结果
        all_results.append({
            'round': round_idx,
            'params': params,
            'best_score': best_score,
            'best_epoch': best_epoch,
            'final_loss': final_loss,
            'early_stopped': early_stopped,
            'model_name': model_name
        })
        
        # 更新历史
        if best_score > best_overall_score:
            best_overall_score = best_score
        
        prev_score = best_score
        
        print(f"\nRound {round_idx} 完成: score={best_score:.4f}, epoch={best_epoch}, trend={trend}")
    
    # 输出总结
    print("\n" + "="*60)
    print("V2 自动调优完成！")
    print("="*60)
    print(f"\n实验结果保存在: {exp_dir}")
    print(f"实验日志: {os.path.join(exp_dir, 'tuning_log.md')}")
    print(f"\n最佳模型得分: {best_overall_score:.4f}")
    
    # 打印Top3模型
    sorted_results = sorted(all_results, key=lambda x: x['best_score'], reverse=True)
    print("\nTop 3 模型:")
    for i, res in enumerate(sorted_results[:TOP_K_BEST]):
        print(f"  {i+1}. {res['model_name']} (score={res['best_score']:.4f})")
    
    # 【关键】保存最佳模型为 Pairwise_Golden_V2_Ultimate
    if len(sorted_results) > 0 and sorted_results[0]['best_score'] > 0:
        best_result = sorted_results[0]
        if best_result['model_name']:
            ultimate_path = '/app/models/Pairwise_Golden_V2_Ultimate.pth'
            source_path = os.path.join(exp_dir, best_result['model_name'], 'best_model.pth')
            if os.path.exists(source_path):
                shutil.copy(source_path, ultimate_path)
                print(f"\n🏆 最佳模型已保存为: {ultimate_path}")
    
    # 生成 TUNING_REPORT_V2.md
    report_path = os.path.join(exp_dir, 'TUNING_REPORT_V2.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# V2 超参调优报告\n\n")
        f.write(f"## 实验时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 实验配置\n\n")
        f.write(f"- NUM_ROUNDS: {NUM_ROUNDS}\n")
        f.write(f"- PARAM_GRID: {PARAM_GRID}\n\n")
        f.write(f"## 结果排行榜\n\n")
        f.write("| 排名 | Round | LR | Dropout | Margin | d_model | layers | Score | Epoch |\n")
        f.write("|------|--------|-----|---------|--------|---------|--------|-------|-------|\n")
        for i, res in enumerate(sorted_results):
            params = res['params']
            f.write(f"| {i+1} | {res['round']} | {params['learning_rate']:.1e} | {params['dropout']} | {params['margin']} | {params['d_model']} | {params['num_layers']} | {res['best_score']:.4f} | {res['best_epoch']} |\n")
        f.write(f"\n## 最佳参数组合\n\n")
        if len(sorted_results) > 0:
            best = sorted_results[0]
            f.write(f"```\n")
            for k, v in best['params'].items():
                f.write(f"{k}: {v}\n")
            f.write(f"best_score: {best['best_score']:.4f}\n")
            f.write(f"best_epoch: {best['best_epoch']}\n")
            f.write(f"```\n")
    print(f"📄 调优报告已保存: {report_path}")
    
    return exp_dir, all_results


if __name__ == '__main__':
    main()
