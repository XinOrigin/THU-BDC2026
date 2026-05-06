# 第一阶段重构方案

> **时间**: 2026-05-04 03:33  
> **状态**: 待审批后实施  
> **触发原因**: auto_tune.py 在"特征工程 88%"阶段 KeyError 崩溃，根因定位为截面排名特征计算位置错误

---

## 一、问题根因分析

### 1.1 当前错误调用链

```
train.py:_preprocess_common()
│
├─ for group in df.groupby('股票代码'):  # 逐只股票循环
│   └─ feature_engineer(group)           # 单只股票数据
│       └─ engineer_features_158plus39(df_copy)
│           └─ add_cross_sectional_ranks(df_final)  ← BUG: 单股数据
│               └─ df.groupby(date_col)[col].rank(pct=True)
│                   # date_col 同一日期只有1条记录，rank无意义
```

**Bug 定位**: [`add_cross_sectional_ranks()`](THU-BDC2026/code/src/utils.py:56) 在 [`engineer_features_158plus39()`](THU-BDC2026/code/src/utils.py:12) 的第 51 行被调用，但此时数据是**单只股票**的时间序列，而非**当日全市场**。

### 1.2 错误后果

| 问题 | 说明 |
|------|------|
| **截面排名变时序排名** | `成交量_rank` 本应是"当天在所有股票中的排名"，实际变成"该股票历史上的成交量分位数" |
| **特征维度丢失** | 模型无法感知市场结构（哪些是热门股、冷门股） |
| **KeyError 崩溃** | [`feature_cloums_map['158+39']`](THU-BDC2026/code/src/train.py:32) 包含 `成交量_rank`，但该列在单股处理后不存在于期望位置 |

### 1.3 特征分类正确性验证

```
时序特征（Time-series）: 在单股 groupby 内部计算
  - 158 个 Alpha 因子（talib 计算，基于该股历史）
  - 39 个技术指标（RSI、MACD、KDJ 等）
  - 这类特征依赖于个股历史，技术上可在单股循环内计算

截面特征（Cross-sectional）: 必须在全市场 concat 后计算
  - 成交量_rank、成交额_rank、涨跌幅_rank、换手率_rank
  - 涨跌幅_rank_change、成交量_rank_change、换手率_rank_change
  - 这类特征描述"当天该股票在市场中的相对位置"
```

---

## 二、重构方案

### 2.1 核心修改：移除 `engineer_features_158plus39()` 中的截面排名调用

**文件**: `THU-BDC2026/code/src/utils.py`

**修改前** (第 50-51 行):
```python
# 6. 【新增】截面排名特征 - 让模型感知市场结构
df_final = add_cross_sectional_ranks(df_final)

return df_final
```

**修改后**:
```python
# 6. 【注意】截面排名特征不再在此处计算
# 截面排名特征需要在全市场 concat 后统一计算，见 train.py:_preprocess_common()

return df_final
```

### 2.2 核心修改：在 concat 后统一计算截面排名

**文件**: `THU-BDC2026/code/src/train.py`

**修改位置**: [`_preprocess_common()`](THU-BDC2026/code/src/train.py:58) 函数

**修改前** (第 74-78 行):
```python
processed_list = []
for group in tqdm(groups, desc=desc):
    processed_list.append(feature_engineer(group))

processed = pd.concat(processed_list).reset_index(drop=True)
```

**修改后**:
```python
processed_list = []
for group in tqdm(groups, desc=desc):
    processed_list.append(feature_engineer(group))

processed = pd.concat(processed_list).reset_index(drop=True)

# 【重构】在 concat 后统一计算截面排名特征（全市场数据）
# 此时所有股票数据已合并，可以正确计算截面排名
from utils import add_cross_sectional_ranks
processed = add_cross_sectional_ranks(processed)
```

### 2.3 脏数据剔除时机调整

**当前流程（错误）**:
```
单股特征工程 → concat → 构建label → 脏数据剔除
```

**重构后流程（正确）**:
```
单股特征工程 → concat → 计算截面排名 → 构建label → 脏数据剔除
```

**脏数据剔除规则**:
- `open_t1 < 1e-4`: 开盘价过小，收益率会极端化
- `label` 为 NaN: 无法计算损失
- 涨停/跌停/停牌标记: 特殊市场状态，不参与训练

**具体修改**: [`_build_label_and_clean()`](THU-BDC2026/code/src/train.py:42) 保持不变，脏数据剔除逻辑维持原位（在 concat + 截面排名之后）。

---

## 三、张量形状推演

### 3.1 配置参数

| 参数 | 值 |
|------|-----|
| `sequence_length` | 50 |
| `batch_size` | 8 (physical) |
| `accumulation_steps` | 4 |
| `effective_batch_size` | 32 |
| `feature_num` | `158+39` = 197 |
| 截面排名特征 | 7 (`成交量_rank`, `成交额_rank`, `涨跌幅_rank`, `换手率_rank`, `涨跌幅_rank_change`, `成交量_rank_change`, `换手率_rank_change`) |
| **总特征数** | **204** |

### 3.2 单个样本张量形状

```
输入: (num_stocks, sequence_length, features)
    = (N, 50, 204)

其中 N = 当日有完整50天历史 + 有未来5天数据的股票数
    ≈ 150~300（沪深300成分股）
```

### 3.3 Batch 张量形状（collate_fn 输出）

```
collate_fn 输入: list of {sequences: Tensor(N_i, 50, 204), ...}
             batch_size = 8

collate_fn 输出:
{
    'sequences':  Tensor(8, max_stocks, 50, 204),   # padded
    'targets':     Tensor(8, max_stocks),
    'relevance':   Tensor(8, max_stocks),
    'stock_indices': Tensor(8, max_stocks),
    'masks':       Tensor(8, max_stocks)
}

max_stocks = max(N_i for i in batch)
           ≈ 300（padding 用 0 填充）
```

### 3.4 Cross-sectional Normalization 维度变化

```python
# collate_fn 第 374-389 行
stacked_sequences = torch.stack(padded_sequences)  # (8, max_stocks, 50, 204)

# Reshape for normalization
batch_size, max_stocks, seq_len, feature_dim = stacked_sequences.shape
# batch_size=8, max_stocks≈300, seq_len=50, feature_dim=204

reshaped = stacked_sequences.permute(0, 2, 1, 3).reshape(batch_size * seq_len, max_stocks, feature_dim)
# (8*50, 300, 204) = (400, 300, 204)

cross_mean = reshaped.mean(dim=1, keepdim=True)  # (400, 1, 204)
cross_std = reshaped.std(dim=1, keepdim=True)    # (400, 1, 204)

normalized_reshaped = (reshaped - cross_mean) / cross_std  # (400, 300, 204)
normalized = normalized_reshaped.reshape(8, 50, 300, 204).permute(0, 2, 1, 3)
# (8, 300, 50, 204) - 恢复原始维度顺序
```

### 3.5 GPU 显存估算（batch_size=8, accumulation_steps=4）

| 组件 | 计算 | 显存占用 |
|------|------|---------|
| 输入张量 | `8 × 300 × 50 × 204 × 4 bytes` | ~9.2 MB |
| 梯度（反向传播） | 与输入同尺寸 | ~9.2 MB |
| 模型参数（d_model=256） | 约 15M 参数 × 4 bytes | ~60 MB |
| 梯度缓存（Adam） | 2 × 模型参数 | ~120 MB |
| 注意力矩阵 | `batch × seq_len × nhead × seq_len × 4bytes` | ~50 MB |
| **总计** | | **~250 MB** |

> ✅ 可在 12GB GPU 上稳定运行

---

## 四、实施步骤

### Step 1: 修改 `utils.py` — 移除截面排名调用
```python
# utils.py 第 50-51 行，删除 add_cross_sectional_ranks 调用
# 仅保留 return df_final
```

### Step 2: 修改 `train.py` — 在 concat 后添加截面排名计算
```python
# train.py _preprocess_common() 第 78 行后添加
processed = add_cross_sectional_ranks(processed)
```

### Step 3: 验证特征数量一致性
```python
# train.py 第 32-34 行 feature_cloums_map['158+39']
# 确认 204 个特征（197 + 7）与实际计算一致
```

### Step 4: 运行 dry_run_test.py 验证流程

### Step 5: 重新启动 auto_tune.py

---

## 五、风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 截面排名计算成为新瓶颈 | concat 后数据量约 3000万行，groupby 可能慢 | 可选：先按日期分组，再计算排名，避免全量 groupby |
| 内存峰值翻倍 | concat 后数据量翻倍 | 分批次处理，或使用 chunked groupby |
| 破坏现有训练流程 | 可能引入未知问题 | Step 4 dry_run 验证 |

---

## 六、预期收益

| 收益 | 说明 |
|------|------|
| **修复 KeyError 崩溃** | 根因修复，auto_tune 可正常运行 |
| **特征语义正确** | 截面排名真正描述市场结构 |
| **模型表达能力提升** | 模型可感知"今天是热门股/冷门股" |
| **避免过拟合** | 时序 rank 和截面 rank 分离，减少噪声 |

---

**请审批后实施上述修改。**