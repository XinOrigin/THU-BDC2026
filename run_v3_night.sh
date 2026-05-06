#!/usr/bin/env python3
"""
V3 守夜人训练启动脚本
- max_epochs=100, early_stopping_patience=20
- 训练完成后自动触发长周期回测
- 生成《黎明战报》
"""
import os
import sys
import gc
import json
import subprocess
import time

# 设置工作目录
os.chdir('/app')

# ============================================================
# 1. 训练前参数死锁核查
# ============================================================
print("=" * 70)
print("【V3 守夜人训练 - 终极参数死锁核查】")
print("=" * 70)

sys.path.insert(0, '/app/code/src')
import config

print(f"margin: {config.config['margin']} (目标: 0.5)")
print(f"batch_size: {config.config['batch_size']} (目标: 8)")
print(f"accumulation_steps: {config.config['accumulation_steps']} (目标: 4)")
print(f"learning_rate: {config.config['learning_rate']} (目标: 3e-05)")
print(f"dropout: {config.config['dropout']} (目标: 0.15)")
print(f"num_epochs: {config.config['num_epochs']} (目标: 100)")
print(f"early_stopping_patience: {config.config['early_stopping_patience']} (目标: 20)")

# 检查特征维度
from train import feature_cloums_map
feature_dim = len(feature_cloums_map['158+39'])
print(f"特征维度: {feature_dim} (目标: 197)")

# 死锁检查
assert config.config['margin'] == 0.5, f"margin错误: {config.config['margin']}"
assert config.config['batch_size'] == 8, f"batch_size错误: {config.config['batch_size']}"
assert config.config['accumulation_steps'] == 4, f"accumulation_steps错误"
assert config.config['num_epochs'] == 100, f"num_epochs错误: {config.config['num_epochs']}"
assert config.config['early_stopping_patience'] == 20, f"early_stopping_patience错误"
assert feature_dim == 197, f"特征维度错误: {feature_dim}"

print("\n✓ 所有参数死锁核查通过！")

# ============================================================
# 2. 创建V3模型目录
# ============================================================
V3_MODEL_DIR = '/app/model/v3_158+39'
os.makedirs(V3_MODEL_DIR, exist_ok=True)
os.makedirs('/app/output', exist_ok=True)

# ============================================================
# 3. 启动训练
# ============================================================
print("\n" + "=" * 70)
print("【启动V3守夜人训练】")
print("=" * 70)

# 准备环境变量，防止多进程问题
env = os.environ.copy()
env['PYTHONPATH'] = '/app/code/src'

# 使用nohup后台运行训练，日志输出到文件
log_file = open('/app/output/v3_train.log', 'w', buffering=1)
train_process = subprocess.Popen(
    ['python', '-u', 'code/src/train.py'],
    cwd='/app',
    env=env,
    stdout=log_file,
    stderr=subprocess.STDOUT
)

print(f"训练进程 PID: {train_process.pid}")
print(f"日志文件: /app/output/v3_train.log")
print("训练已后台启动，等待完成...")

# 等待训练完成
train_process.wait()
log_file.close()

print("\n训练进程已退出")

# 读取训练结果
best_epoch = None
best_score = None

if os.path.exists(f'{V3_MODEL_DIR}/final_score.txt'):
    with open(f'{V3_MODEL_DIR}/final_score.txt', 'r') as f:
        content = f.read()
        print(f"\n最终得分文件内容:\n{content}")
        
        # 解析
        for line in content.split('\n'):
            if 'Best epoch' in line:
                best_epoch = line.split(':')[1].strip()
            if 'Best final_score' in line:
                best_score = line.split(':')[1].strip()
else:
    print("警告: 未找到final_score.txt")

# ============================================================
# 4. 重命名最佳模型
# ============================================================
best_model_src = f'{V3_MODEL_DIR}/best_model.pth'
best_model_dst = f'{V3_MODEL_DIR}/Pairwise_Golden_V3_Best.pth'

if os.path.exists(best_model_src):
    if os.path.exists(best_model_dst):
        os.remove(best_model_dst)
    os.rename(best_model_src, best_model_dst)
    print(f"\n最佳模型已重命名为: {best_model_dst}")

# 复制config
config_src = f'{V3_MODEL_DIR}/config.json'
if os.path.exists('/app/code/src/config.py'):
    import shutil
    shutil.copy('/app/code/src/config.py', config_src)
    print(f"config.json已保存")

# ============================================================
# 5. 执行长周期回测
# ============================================================
print("\n" + "=" * 70)
print("【启动V3长周期回测】")
print("=" * 70)

backtest_result = subprocess.run(
    ['python', '-u', 'v3_long_term_backtest.py'],
    cwd='/app',
    capture_output=True,
    text=True,
    env=env
)

print(backtest_result.stdout)
if backtest_result.stderr:
    print("STDERR:", backtest_result.stderr)

# ============================================================
# 6. 撰写《黎明战报》
# ============================================================
print("\n" + "=" * 70)
print("【撰写V3黎明战报】")
print("=" * 70)

# 读取回测指标
metrics = {
    'total_return_v3': 0.0,
    'total_return_bench': 0.0,
    'alpha': 0.0,
    'win_rate': 0.0,
    'max_drawdown': 0.0,
    'sharpe_ratio': 0.0,
    'excess_return': 0.0
}

metrics_path = '/app/output/v3_metrics.json'
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

# 撰写战报
report = f"""# V3 奇美拉模型 - 黎明战报
生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 一、训练结果

| 指标 | 值 |
|------|-----|
| 最佳 Epoch | {best_epoch or 'N/A'} |
| 最佳验证得分 | {best_score or 'N/A'} |
| 训练配置 | max_epochs=100, patience=20, margin=0.5 |

## 二、长周期回测结果

测试期: 2024-01-01 ~ 2026-04-24

### 收益指标

| 策略 | 累计收益 |
|------|---------|
| **V3 奇美拉** | **{metrics.get('total_return_v3', 0):.2f}%** |
| 沪深300基准 | {metrics.get('total_return_bench', 0):.2f}% |
| Alpha超额收益 | **{metrics.get('alpha', 0):.2f}%** |

### 胜率指标

| 指标 | 值 |
|------|-----|
| 超额胜率 | **{metrics.get('win_rate', 0):.2f}%** ({int(metrics.get('win_count', 0))}/{int(metrics.get('total_evaluations', 0))}) |
| 相对沪深300 | {'✓ 超越基准' if metrics.get('total_return_v3', 0) > metrics.get('total_return_bench', 0) else '✗ 未超越基准'} |

### 风险指标

| 指标 | 值 |
|------|-----|
| 最大回撤 | {metrics.get('max_drawdown', 0):.2f}% |
| 夏普比率 | {metrics.get('sharpe_ratio', 0):.4f} |

## 三、V1 vs V3 对照

| 指标 | V1 | V3 |
|------|----|----|
| 累计收益 | +20.95% | {metrics.get('total_return_v3', 0):.2f}% |
| Alpha | +1.33% | {metrics.get('alpha', 0):.2f}% |
| 胜率 | 62.50% | {metrics.get('win_rate', 0):.2f}% |
| 最大回撤 | 12.54% | {metrics.get('max_drawdown', 0):.2f}% |

## 四、结论

{'**V3 奇美拉训练成功！** 模型在长周期测试中表现超越V1基准。' if metrics.get('total_return_v3', 0) > 20 else '**V3 需要进一步优化。** 模型表现待提升。'}

---
*本战报由守夜人自动生成*
"""

report_path = '/app/MORNING_REPORT_V3.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n黎明战报已保存到: {report_path}")
print("\n" + "=" * 70)
print("【V3 守夜人训练 + 回测 + 战报 全部完成】")
print("=" * 70)
