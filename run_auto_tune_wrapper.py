#!/usr/bin/env python3
"""Wrapper to run auto_tune in Docker"""
import sys
sys.path.insert(0, '/app/code/src')
from auto_tune import main
main()