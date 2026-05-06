# V3 Config - 融合 V1 数据管道 + V2 网络稳定性
# 核心改变：废除截面归一化，恢复 margin=0.5，锁定197维特征

sequence_length = 50
feature_num = '158+39'
temperature = 1.0

config = {
    'sequence_length': sequence_length,
    'd_model': 256,           # V2 架构
    'nhead': 16,              # V2 架构
    'num_layers': 3,          # V2 架构
    'dim_feedforward': 512,

    # V2 防OOM设计
    'batch_size': 8,          # V2 降至 8 防OOM
    'accumulation_steps': 4, # V2 梯度累加，effective batch=32

    # V2 网络稳定性
    'dropout': 0.15,          # V2 实测不过拟合

    # V1 回归
    'margin': 0.5,            # 【关键】恢复V1默认值，替换V2的0.05
    'learning_rate': 3e-05,   # V2 实测稳定值

    'num_epochs': 100,        # V3 充分训练（守夜人长跑）
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    'pairwise_weight': 1,
    'base_weight': 1.0,
    'top5_weight': 2.0,

    'early_stopping_patience': 20,  # 守夜人长跑防护
    'warmup_epochs': 3,             # V3 增加预热

    'output_dir': f'./models/v3_{feature_num}',
    'data_path': './data',
}