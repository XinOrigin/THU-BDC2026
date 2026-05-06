#!/usr/bin/env python3
"""测试截面排名特征"""
import sys
sys.path.insert(0, '/app/code/src')

import pandas as pd
from utils import add_cross_sectional_ranks

# 简单测试
test_df = pd.DataFrame({
    '日期': ['2021-09-28'] * 5 + ['2021-09-29'] * 5,
    'instrument': list(range(5)) + list(range(5)),
    '成交量': [100, 200, 300, 400, 500] * 2,
    '涨跌幅': [1, 2, 3, 4, 5] * 2,
})
test_df['日期'] = pd.to_datetime(test_df['日期'])

result = add_cross_sectional_ranks(test_df)
print('新增特征列:', [c for c in result.columns if '_rank' in c])
print()
print('2021-09-28 成交量_rank:')
print(result[result['日期']=='2021-09-28'][['instrument', '成交量', '成交量_rank']])