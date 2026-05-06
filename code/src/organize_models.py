#!/usr/bin/env python3
"""
整理auto_tune生成的模型，将每个temp_round的结果整理成规范文件夹结构

规范结构：
model/
└── Tuning_Results/
    ├── Round01_lr4e-05_drop0.1_temp2.0/
    │   ├── config.json
    │   ├── best_model.pth
    │   └── final_score.txt
    ├── Round02_lr1e-05_drop0.2_temp1.0/
    │   └── ...
    └── tuning_summary.md
"""

import os
import json
import shutil

def organize_tuning_results():
    base_model_dir = '/app/model'
    source_dir = base_model_dir
    
    # 创建目标目录
    target_dir = os.path.join(base_model_dir, 'Tuning_Results')
    os.makedirs(target_dir, exist_ok=True)
    
    # 遍历所有temp_round目录
    temp_rounds = []
    for item in os.listdir(source_dir):
        if item.startswith('temp_round_'):
            temp_rounds.append(item)
    
    temp_rounds.sort(key=lambda x: int(x.split('_')[-1]))
    
    summary_data = []
    
    for round_folder in temp_rounds:
        round_path = os.path.join(source_dir, round_folder)
        round_num = int(round_folder.split('_')[-1])
        
        # 读取config
        config_path = os.path.join(round_path, 'config.json')
        if not os.path.exists(config_path):
            print(f"跳过 {round_folder}，未找到config.json")
            continue
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        lr = config.get('learning_rate', 4e-5)
        drop = config.get('dropout', 0.2)
        temp = config.get('temperature', 1.0)
        
        # 创建目标文件夹名称
        folder_name = f"Round{round_num:02d}_lr{lr}_drop{drop}_temp{temp}"
        model_folder = os.path.join(target_dir, folder_name)
        
        if os.path.exists(model_folder):
            print(f"文件夹已存在，跳过: {folder_name}")
            continue
        
        os.makedirs(model_folder)
        
        # 复制文件
        files_to_copy = ['config.json', 'best_model.pth']
        for fname in files_to_copy:
            src = os.path.join(round_path, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(model_folder, fname))
        
        # 复制scaler如果存在
        scaler_path = os.path.join(round_path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            shutil.copy(scaler_path, os.path.join(model_folder, 'scaler.pkl'))
        
        # 复制result.csv如果存在
        result_path = os.path.join(round_path, 'result.csv')
        if os.path.exists(result_path):
            shutil.copy(result_path, os.path.join(model_folder, 'result.csv'))
        
        print(f"已整理: {folder_name}")
        summary_data.append({
            'round': round_num,
            'lr': lr,
            'dropout': drop,
            'temperature': temp,
            'folder': folder_name
        })
    
    # 生成汇总文件
    summary_path = os.path.join(target_dir, 'tuning_summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# 自动调优结果汇总\n\n")
        f.write("| 序号 | 学习率 | Dropout | Temperature | 文件夹 |\n")
        f.write("|------|--------|---------|------------|--------|\n")
        for item in summary_data:
            f.write(f"| {item['round']} | {item['lr']} | {item['dropout']} | {item['temperature']} | {item['folder']} |\n")
    
    print(f"\n整理完成! 共 {len(summary_data)} 个模型")
    print(f"结果保存在: {target_dir}")
    print(f"汇总文件: {summary_path}")

if __name__ == '__main__':
    organize_tuning_results()
