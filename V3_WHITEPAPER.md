# V3 终极重构白皮书

## 一、实验结论总结

### V1 vs V2 跨周期对照结果 (2024-01-02 ~ 2026-04-24)

| 指标 | V1 (Pairwise_Golden) | V2 (Round08) | 沪深300基准 |
|------|---------------------|--------------|-------------|
| **累计收益** | **+20.95%** | -3.79% | +19.62% |
| **Alpha (vs基准)** | **+1.11%** | -19.56% | -- |
| **正收益胜率** | **~46%** | 45.9% | -- |
| **超额基准胜率** | **~50%** | 45.0% | -- |
| **夏普比率** | **~1.0** | -0.02 | 1.053 |
| **最大回撤** | **~23%** | 26.58% | 23.62% |

**结论**: V1 以显著优势碾压 V2！

---

## 二、根本原因诊断

### V2 失败根因：截面归一化抹平绝对动量信号

V2 在 `collate_fn` 中对每个时间截面做 Z-score 归一化：
```python
batch_mean = batch_seq.mean(dim=0, keepdim=True)
batch_std = batch_seq.std(dim=0, keepdim=True) + 1e-8
batch_seq_norm = (batch_seq - batch_mean) / batch_std
```

**问题**：
- 抹平了不同股票的绝对价格水平差异
- 在大盘普涨时，V2只知道"哪只股票相对最强"，但不知道"市场整体在上涨"
- 丢失了绝对动量信号 → 模型只能做相对排序，无法感知市场方向

### 截面排名特征的副作用

V2 新增的 7 个截面排名特征（成交量_rank, 成交额_rank 等）在长周期中成为噪声：
- 排名只反映"当天热门程度"，不反映"第二天继续涨"
- 在沪深300普涨时期，热门股往往已经超买，反而下跌

---

## 三、V3 缝合方案

### 核心原则
> **V3 = V1的数据管道（保留绝对信号）+ V2的网络稳定性 + V1的margin默认值**

### 各组件选择

| 组件 | V3 选择 | 来源 |
|------|---------|------|
| **特征维度** | **197维** | V1 纯净版 |
| **特征列表** | instrument + 158 Alpha + 39 Tech | V1 标准 |
| **截面排名特征** | ❌ **禁用** | 废除V2的7个rank列 |
| **归一化方式** | **RobustScaler（绝对值）** | V1 方式，废除DataLoader截面Z-score |
| **网络结构** | **V2架构** | d_model=256, nhead=16, num_layers=3 |
| **LayerNorm** | ✅ 保留 | V2 稳定性设计 |
| **batch_size** | **8** | V2 防OOM设计 |
| **accumulation_steps** | **4** | V2 防OOM设计 |
| **learning_rate** | **3e-05** | V2 实测稳定值 |
| **dropout** | **0.15** | V2 实测不 过拟合 |
| **Loss函数** | **PairwiseRankingLoss** | 保留排序能力 |
| **margin** | **0.5** | 恢复V1默认值（替换V2的0.05） |

### V3 配置完整 JSON

```json
{
    "sequence_length": 50,
    "feature_num": "158+39",
    "d_model": 256,
    "nhead": 16,
    "num_layers": 3,
    "dim_feedforward": 512,
    "batch_size": 8,
    "num_epochs": 30,
    "dropout": 0.15,
    "max_grad_norm": 5.0,
    "pairwise_weight": 1,
    "base_weight": 1.0,
    "top5_weight": 2.0,
    "early_stopping_patience": 10,
    "warmup_epochs": 3,
    "data_path": "/app/data",
    "accumulation_steps": 4,
    "learning_rate": 3e-05,
    "margin": 0.5,
    "output_dir": "./models/v3"
}
```

---

## 四、关键修改点

### 1. 数据管道（train.py）

**恢复 V1 的 197 维特征列表**：
```python
feature_cloums_map = {
    '158+39': [
        'instrument',  # 保留instrument
        '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        # ... 158 Alpha因子 ...
        # ... 39 Tech指标 ...
        # 【无】截面排名特征
    ]
}
```

**禁用 `add_cross_sectional_ranks` 调用**（或确保在concat后调用但不加入特征列表）

### 2. collate_fn（废除截面Z-score）

```python
def collate_fn(batch):
    # 【关键修改】废除截面归一化，直接使用scaler处理后的绝对值
    # 不再做 batch_seq_norm = (batch_seq - mean) / std
    ...
```

### 3. margin 恢复 0.5

```python
criterion = PairwiseRankingLoss(margin=config['margin'], k=5)
# V3 margin = 0.5（恢复V1默认值）
```

---

## 五、预期效果

| 指标 | 预期 |
|------|------|
| 累计收益 | ~+20% (V1水平) |
| Alpha | 正（跑赢沪深300基准） |
| 夏普比率 | >0.8 |
| 最大回撤 | <25% |

**核心信心**：保留 V2 网络稳定性设计 + 恢复 V1 绝对信号 → 应该是 V1 的稳健提升版。