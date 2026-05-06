#!/bin/bash
# 回炉重造训练脚本：使用彻底移除 instrument 特征的干净数据训练 Golden Model
# Golden Config: lr=8e-5, dropout=0.3, margin=0.5

set -e

echo "============================================================"
echo "回炉重造：移除 instrument 特征，重新训练 Golden Model"
echo "============================================================"

MODEL_NAME="Golden_NoInstrument_$(date +%Y%m%d_%H%M%S)"
MODEL_DIR="/app/model/${MODEL_NAME}"

echo "创建模型目录: ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

# 执行训练（使用 Golden Config，这些参数已锁定在 config.py 中）
echo ""
echo "开始训练..."
echo "关键参数: lr=8e-5, dropout=0.3, margin=0.5, d_model=256, num_layers=3"

cd /app

python code/src/train.py 2>&1 | tee "${MODEL_DIR}/train_log.txt"

echo ""
echo "============================================================"
echo "训练完成！"
echo "============================================================"

# 查找最佳模型
BEST_MODEL=$(find "${MODEL_DIR}" -name "best_model*.pth" | head -1)
if [ -z "$BEST_MODEL" ]; then
    echo "错误：未找到训练好的模型文件"
    exit 1
fi

echo "最佳模型: ${BEST_MODEL}"

# 读取训练结果
FINAL_SCORE=$(cat "${MODEL_DIR}/final_score.txt" 2>/dev/null || echo "N/A")
echo "训练分数: ${FINAL_SCORE}"

# 保存模型路径供后续使用
echo "${MODEL_DIR}" > /app/output/golden_model_path.txt
echo "${BEST_MODEL}" >> /app/output/golden_model_path.txt

echo ""
echo "模型已保存到: ${MODEL_DIR}"
echo "接下来请运行无重叠 T+5 回测"