#!/usr/bin/env python3
"""
回炉重造脚本：使用彻底移除 instrument 特征的干净数据，训练 Golden Model
然后执行无重叠 T+5 回测

Golden Config: lr=8e-5, dropout=0.3, margin=0.5, d_model=256, num_layers=3
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime

# 确保在 /app 目录
os.chdir('/app')
sys.path.insert(0, '/app/code/src')

def run_cmd(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f">>> {desc}")
    print(f"CMD: {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0

def main():
    model_name = f"Golden_NoInstrument_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_dir = f"/app/model/{model_name}"
    
    print(f"创建模型目录: {model_dir}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Step 1: 执行单次训练，使用 Golden Config
    # 关键参数: lr=8e-5, dropout=0.3, margin=0.5, d_model=256, num_layers=3
    train_cmd = (
        "cd /app && "
        "python code/src/train.py "
        "--epochs 25 "
        "--learning_rate 8e-5 "
        "--dropout 0.3 "
        "--margin 0.05 "
        "--d_model 256 "
        "--num_layers 3 "
        "--batch_size 256 "
        "--warmup_epochs 5 "
        f"--output_dir {model_dir} "
        "--device cuda "
        "--seed 42 "
        "--feature_num 158+39 "
        "--early_stopping_patience 15 "
        "--weight_decay 0.01 "
        "--clip_grad 1.0 "
        "2>&1 | tee /app/output/retrain_log.txt"
    )
    
    success = run_cmd(train_cmd, f"回炉重造训练 - Golden Config (lr=8e-5, dropout=0.3, margin=0.5)")
    
    if not success:
        print("训练失败！")
        sys.exit(1)
    
    # 查找最佳模型
    best_model_path = None
    best_score = -999
    for f in os.listdir(model_dir):
        if f.startswith('best_model') and f.endswith('.pth'):
            # 尝试从文件名解析 score
            if 'score' in f:
                try:
                    score_str = f.split('score')[1].split('_')[0]
                    score = float(score_str)
                    if score > best_score:
                        best_score = score
                        best_model_path = os.path.join(model_dir, f)
                except:
                    pass
            else:
                if best_model_path is None:
                    best_model_path = os.path.join(model_dir, f)
    
    if best_model_path is None:
        # 查找任意 .pth 文件
        for f in os.listdir(model_dir):
            if f.endswith('.pth'):
                best_model_path = os.path.join(model_dir, f)
                break
    
    print(f"\n最佳模型: {best_model_path}")
    print(f"最佳分数: {best_score}")
    
    # Step 2: 创建无重叠 T+5 回测脚本并执行
    print("\n" + "="*60)
    print(">>> 执行无重叠 T+5 回测")
    print("="*60)
    
    # 读取训练日志获取 final_score
    final_score = 0.0
    try:
        with open(os.path.join(model_dir, 'final_score.txt'), 'r') as f:
            final_score = float(f.read().strip())
    except:
        pass
    
    print(f"模型训练最终分数: {final_score}")
    print(f"模型保存路径: {model_dir}")
    print(f"模型文件: {os.path.basename(best_model_path)}")
    
    print("\n请手动执行回测分析:")
    print(f"  python /app/walkforward_noinstrument.py --model_dir {model_dir}")

if __name__ == '__main__':
    main()