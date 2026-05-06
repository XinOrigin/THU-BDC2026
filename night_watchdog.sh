#!/bin/bash
###############################################################################
# Night Watchdog Script - 夜间自动调优看门狗
#
# 功能:
# 1. 捕获 auto_tune.py 的 Exit Code
# 2. 崩溃后执行环境清洗（杀进程、释放内存）
# 3. 自动拉起下一轮训练
# 4. 完整日志记录
#
# 使用方式: nohup bash night_watchdog.sh &
###############################################################################

set -o pipefail

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_TUNE_SCRIPT="/app/auto_tune.py"
LOG_FILE="/app/output/auto_tune_v2_night.log"
CRASH_LOG="/app/NIGHT_CRASH_LOG.md"
WRAPPER_SCRIPT="/app/run_auto_tune_wrapper.py"
PYTHON_PATH="/app/code/src"

# 初始化崩溃日志
init_crash_log() {
    if [ ! -f "$CRASH_LOG" ]; then
        cat > "$CRASH_LOG" << 'EOF'
# 夜间崩溃日志 Night Crash Log

## 格式
- 时间 | Round N | 错误类型 | 错误信息摘要

---

EOF
        fi
}

# 记录崩溃
log_crash() {
    local round_num="$1"
    local error_type="$2"
    local error_msg="$3"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "## [$timestamp] Round $round_num | $error_type" >> "$CRASH_LOG"
    echo '```' >> "$CRASH_LOG"
    echo "$error_msg" >> "$CRASH_LOG"
    echo '```' >> "$CRASH_LOG"
    echo "" >> "$CRASH_LOG"
}

# 环境清洗
cleanup_environment() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 执行环境清洗..." | tee -a "$LOG_FILE"
    
    # 杀残留Python进程
    docker exec thu-bdc2026-app-1 bash -c "pkill -9 python 2>/dev/null || true"
    
    # 等待系统释放RAM
    sleep 10
    
    # 清理临时文件
    docker exec thu-bdc2026-app-1 bash -c "rm -rf /app/models/temp_round_* 2>/dev/null || true"
    docker exec thu-bdc2026-app-1 bash -c "rm -rf /app/temp 2>/dev/null || true"
    
    # 再次确认进程已清除
    docker exec thu-bdc2026-app-1 bash -c "ps aux | grep python | grep -v grep || echo 'No python processes'"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 环境清洗完成" | tee -a "$LOG_FILE"
}

# 等待Docker恢复
wait_for_docker() {
    local max_wait=300  # 最多等5分钟
    local count=0
    
    while [ $count -lt $max_wait ]; do
        if docker ps --format '{{.Names}}' | grep -q "thu-bdc2026-app-1"; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Docker 已恢复" | tee -a "$LOG_FILE"
            return 0
        fi
        sleep 2
        count=$((count + 2))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 等待 Docker 恢复... ($count/$max_wait)" >> "$LOG_FILE"
    done
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Docker 恢复超时!" | tee -a "$LOG_FILE"
    return 1
}

# 启动auto_tune
start_auto_tune() {
    local attempt=$1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 auto_tune.py (尝试 $attempt)" | tee -a "$LOG_FILE"
    
    # 同步最新代码
    docker cp "$SCRIPT_DIR/code/src/auto_tune.py" thu-bdc2026-app-1:/app/auto_tune.py
    docker cp "$SCRIPT_DIR/code/src/train.py" thu-bdc2026-app-1:/app/code/src/train.py
    docker cp "$SCRIPT_DIR/code/src/utils.py" thu-bdc2026-app-1:/app/code/src/utils.py
    
    # 后台启动
    docker exec -d thu-bdc2026-app-1 bash -c "cd /app && PYTHONPATH=$PYTHON_PATH python $WRAPPER_SCRIPT >> $LOG_FILE 2>&1"
    
    # 等待启动确认
    sleep 5
    
    # 检查进程是否在运行
    if docker exec thu-bdc2026-app-1 bash -c "ps aux | grep -v grep | grep -q 'auto_tune.py\|run_auto_tune_wrapper.py'"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] auto_tune.py 已启动" | tee -a "$LOG_FILE"
        return 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] auto_tune.py 启动失败" | tee -a "$LOG_FILE"
        return 1
    fi
}

###############################################################################
# 主循环
###############################################################################

init_crash_log

echo "============================================================" | tee -a "$LOG_FILE"
echo "夜间看门狗启动 $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# 等待Docker可用
wait_for_docker || exit 1

# 确保输出目录存在
docker exec thu-bdc2026-app-1 bash -c "mkdir -p /app/output"

# 启动第一轮
start_auto_tune 1

# 主监控循环
consecutive_failures=0
max_consecutive_failures=5

while true; do
    # 检查进程是否存活
    if docker exec thu-bdc2026-app-1 bash -c "ps aux | grep -v grep | grep -q 'auto_tune.py\|run_auto_tune_wrapper.py'"; then
        consecutive_failures=0
        # 进程存活，休眠30秒后继续监控
        sleep 30
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 检测到进程死亡" | tee -a "$LOG_FILE"
        
        # 获取退出状态
        local exit_code=$?
        
        # 从日志中提取当前Round信息
        local last_round=$(docker exec thu-bdc2026-app-1 bash -c "grep -o 'Round[0-9]*' $LOG_FILE 2>/dev/null | tail -1" 2>/dev/null || echo "Unknown")
        
        # 提取最后几行错误信息
        local error_snippet=$(docker exec thu-bdc2026-app-1 bash -c "tail -50 $LOG_FILE 2>/dev/null | grep -A5 -B5 'Error\|Exception\|Traceback\|KeyError\|OOM\|CUDA' | tail -20" 2>/dev/null || echo "No error details")
        
        # 记录崩溃
        log_crash "$last_round" "ExitCode:$exit_code" "$error_snippet"
        
        consecutive_failures=$((consecutive_failures + 1))
        
        if [ $consecutive_failures -ge $max_consecutive_failures ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 连续失败次数过多 ($consecutive_failures)，停止自动重启" | tee -a "$LOG_FILE"
            echo "请人工检查系统状态" | tee -a "$LOG_FILE"
            exit 1
        fi
        
        # 环境清洗
        cleanup_environment
        
        # 等待Docker恢复
        wait_for_docker || exit 1
        
        # 重新启动
        start_auto_tune $((consecutive_failures + 1))
    fi
    
    # 每小时打印一次心跳
    local hour=$(date '+%H')
    local minute=$(date '+%M')
    if [ "$minute" = "00" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 看门狗心跳 - 进程运行中" | tee -a "$LOG_FILE"
    fi
done