#!/bin/bash
cd /app
export PYTHONPATH=/app/code/src
nohup /app/.venv/bin/python -c "
import sys
sys.path.insert(0, '/app/code/src')
from auto_tune import main
main()
" > /app/output/auto_tune_v2.log 2>&1 &