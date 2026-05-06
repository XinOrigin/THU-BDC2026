  
---  
  
### [2026-05-02] Docker�����ļ�ͬ�������޸���auto_tune.py·������  
* **�޸ĵ��ļ�**: `THU-BDC2026/docker-compose.yml`, `THU-BDC2026/code/src/auto_tune.py`  
* **���Ĳ���**:   
  * ���⣺volume mapping `./model:/app/model` ָ�򲻴��ڵ�Ŀ¼�������������޷�����ģ���ļ�  
  * �޸�����docker-compose.yml�е�volume��`./model:/app/model`��Ϊ`./models:/app/model`  
  * ͬʱ��auto_tune.py�е���ʱĿ¼·����`./temp_round_{n}`����Ϊ`./models/temp_round_{n}`  
  * ����ʵ�����·����ʹ��`/app/model`��Ϊ����·���������ڣ�  
* **״̬**: [x] ����� 
  
---  
  
### [2026-05-02] ����̽�������������滯�ع�  
* **�޸ĵ��ļ�**: `THU-BDC2026/code/src/utils.py`, `THU-BDC2026/code/src/probe_check.py`  
* **���Ĳ���**:   
  * ̽�����������չ�Ʊ����284�����̼�CV=2.38���ɽ���CV=2.52��ȷ������δ��һ��  
  * �������������������� `add_cross_sectional_ranks()`��Ϊ�ɽ������ɽ���ǵ��������������ӽ���ٷ�λ����  
  * ���������仯�������ǵ���_rank_change���ɽ���_rank_change��������_rank_change  
  * ��ģ�͸�֪�г��ṹ�����ǽ����ɼ���ָ��  
* **״̬**: [x] ����� 

---

### [2026-05-03] 温度扫描完成 - T=0.01 强势胜出

* **修改的文件**: `THU-BDC2026/temp_sweep6.py`

* **核心操作**: 
  * 修复特征维度不匹配：之前报错 203 vs 197，根因是 `add_cross_sectional_ranks` 生成了6个额外排名特征
  * 解决方案：对特征矩阵切片，只取前197列（丢弃最后6个：`成交额_rank`, `涨跌幅_rank`, `换手率_rank`, `涨跌幅_rank_change`, `成交量_rank_change`, `换手率_rank_change`）
  * 使用 LINEARSUM 评估（竞赛官方方法），而非 compound
  * 49次非重叠 T+5 评估结果：

| 方案 | 总线性收益 | 平均单次收益 | 单次最高爆发 | 胜率 |
|------|-----------|-------------|-------------|------|
| Equal(20%) | 0.1546 | 0.003154 | 0.2156 | 40.8% |
| T=1.0 | 0.1551 | 0.003166 | 0.2155 | 40.8% |
| T=0.5 | 0.1557 | 0.003178 | 0.2154 | 40.8% |
| T=0.1 | 0.1607 | 0.003280 | 0.2145 | 40.8% |
| **T=0.01** | **0.2382** | **0.004861** | 0.2041 | **49.0%** |

  * **结论**：T=0.01（赢家通吃策略）以 54% 优势领先总线性收益，胜率也最高

* **状态**: [x] 已完成


---

### [2026-05-03] 比赛提交通用脚本 generate_submission.py 完成

* **修改的文件**: `THU-BDC2026/generate_submission.py`

* **核心操作**: 
  * 编写比赛专用提交流水线，使用 Pairwise_Golden 模型 + T=0.01 温度参数
  * 锁定特征截断逻辑（只取前197列，丢弃最后6个排名特征）
  * 权重四舍五入到4位小数，确保总和严格 <= 1.0
  * 输出格式：`stock_id,weight`（UTF-8编码）
  * 测试运行结果：
    ```
    stock_id,weight
    600000,0.2623
    600010,0.2189
    2475,0.1801
    600009,0.1701
    300274,0.1686
    ```
    权重总和: 1.0 ✓

* **状态**: [x] 已完成


---

### [2026-05-03] 底层除虫修复：截面归一化 + margin调低

* **修改的文件**: `THU-BDC2026/code/src/train.py`, `THU-BDC2026/code/src/config.py`

* **核心修复**:

1. **截面归一化 (Cross-sectional Normalization)**:
   - 问题: 特征在每个时间截面（每天）没有做 Z-score 标准化，成交额等大数值特征主导 Attention 计算
   - 修复: 在 `collate_fn` 中添加截面归一化
     - 对每个日期的股票池，计算每个时间步的特征均值和标准差
     - 对所有股票做 Z-score 标准化：`(x - mean) / (std + 1e-8)`
   - 验证: mean ≈ 0, std ≈ 1.0 ✓

2. **Margin 调低**:
   - 问题: `margin=0.5` 太大，当 logits 差异仅 ~0.01 时，Hinge Loss 几乎总是 0，梯度信号微弱
   - 修复: `margin` 从 0.5 降到 0.05

* **状态**: [x] 已完成修复，待重新训练验证效果

---

### [2026-05-05] V3 代码重组：移除截面归一化 + 恢复 V1 margin=0.5

* **修改的文件**: `THU-BDC2026/code/src/train.py`, `THU-BDC2026/code/src/test.py`, `THU-BDC2026/code/src/config.py`, `THU-BDC2026/dry_run_test.py`

* **核心操作**:
  1. **特征列表降维**：`train.py` 和 `test.py` 中的 `feature_cloums_map['158+39']` 从 203 维删除 7 个截面排名特征（`成交量_rank`, `成交额_rank`, `涨跌幅_rank`, `换手率_rank`, `涨跌幅_rank_change`, `成交量_rank_change`, `换手率_rank_change`），恢复到 197 维
  2. **移除截面 Z-score 标准化**：`train.py` 的 `collate_fn` 中删除了 `stacked_sequences` 的跨股票 Z-score 标准化代码（约 20 行），保留原始绝对动量信号
  3. **移除 add_cross_sectional_ranks 调用**：`train.py` 的 `_preprocess_common()` 中删除了 concat 后调用 `add_cross_sectional_ranks()` 的代码
  4. **更新 V3 配置**：`config.py` 已更新为 V3 配置：
     - `margin=0.5`（恢复 V1 默认值，替换 V2 的 0.05）
     - `batch_size=8`, `accumulation_steps=4`（有效 batch=32）
     - `dropout=0.15`, `learning_rate=3e-05`
  5. **V3 战前彩排脚本**：`dry_run_test.py` 重写为 V3 干跑测试，验证 197 维特征、margin=0.5、无 NaN/Inf

* **V3 架构总结**:
  - V3 = V1 数据管道（197 维原始特征）+ V2 网络稳定性（d_model=256, dropout=0.15, 梯度累加）
  - 关键改变：废除 V2 的截面归一化（曾抹平绝对动量信号，导致长周期失败）
  - margin 从 0.05 恢复到 0.5（V1 原始设置）

* **状态**: [x] 已完成代码重组，全面进入待机状态，等待全量点火指令

---

### [2026-05-05] V3 守夜人全量点火启动

* **修改的文件**: `THU-BDC2026/code/src/config.py`, `THU-BDC2026/v3_long_term_backtest.py`, `THU-BDC2026/run_v3_night.sh`, `THU-BDC2026/docker-compose.yml`

* **核心操作**:
  1. **config.py参数死锁**: `num_epochs=100`, `early_stopping_patience=20`, `margin=0.5`
  2. **创建V3长周期回测脚本**: `v3_long_term_backtest.py` - 测试期2024-01-01~2026-04-24，无截面排名特征
  3. **创建守夜人启动脚本**: `run_v3_night.sh` - 训练→回测→战报全自动
  4. **docker-compose.yml修改**: 挂载整个项目目录，执行`run_v3_night.sh`

* **V3守夜人训练配置**:
  - max_epochs=100, early_stopping_patience=20
  - margin=0.5, batch_size=8, accumulation_steps=4, lr=3e-05, dropout=0.15
  - 特征维度: 197（无截面排名特征）

* **状态**: [x] 已点火，后台运行中，等待明早黎明战报
