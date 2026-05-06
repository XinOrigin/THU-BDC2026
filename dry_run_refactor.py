#!/usr/bin/env python3
"""
Dry Run Test - 干跑查杀测试 (Phase 1 Refactoring Validation)
验证：截面排名特征计算位置修复后，feature_dim=204 的前向传播正常
"""
import torch
import sys
sys.path.insert(0, '/app/code/src')

print('='*60)
print('DRY RUN TEST - Phase1 Refactoring Validation')
print('='*60)

# 1. Import modified modules
import model, config, train
print(f'\n[1] Config d_model: {config.config["d_model"]}')
print(f'    Config dropout: {config.config["dropout"]} (expected 0.15)')

# 2. Create mock batch with feature_dim=204
batch_size = 2
max_stocks = 5
seq_len = 50
feature_dim = 204  # 197 Alpha+Tech + 7 Cross-sectional rank features

mock_batch = []
for i in range(batch_size):
    seq = torch.randn(max_stocks, seq_len, feature_dim) * 10 + 5  # mean=5, std=10
    tgt = torch.randn(max_stocks) * 0.1
    rel = torch.randint(0, 300, (max_stocks,))
    stock_idx = torch.randint(0, 300, (max_stocks,))
    
    mock_batch.append({
        'sequences': seq,
        'targets': tgt,
        'relevance': rel,
        'stock_indices': stock_idx
    })

print(f'\n[2] Mock batch: batch_size={batch_size}, max_stocks={max_stocks}, seq_len={seq_len}, feature_dim={feature_dim}')

# 3. Run collate_fn with cross-sectional normalization
print('\n[3] Running collate_fn...')
try:
    result = train.collate_fn(mock_batch)
    print('    [OK] collate_fn completed')
except Exception as e:
    print(f'    [ERROR] collate_fn: {e}')
    import traceback; traceback.print_exc()
    raise

# 4. Check for NaN/Inf
sequences = result['sequences']
print(f'\n[4] Feature NaN/Inf check:')
print(f'    Shape: {sequences.shape}')
print(f'    has NaN: {torch.isnan(sequences).any()}')
print(f'    has Inf: {torch.isinf(sequences).any()}')
print(f'    mean: {sequences.mean().item():.6f}')
print(f'    std: {sequences.std().item():.6f}')

if torch.isnan(sequences).any() or torch.isinf(sequences).any():
    print('    [FATAL] NaN or Inf detected!')
    raise ValueError('NaN/Inf detected')

# 5. Check cross-sectional normalization
print('\n[5] Cross-sectional normalization check:')
for b in range(batch_size):
    for t in range(min(3, seq_len)):
        slice_ = sequences[b, :, t, :]
        cross_mean = slice_.mean().item()
        cross_std = slice_.std().item()
        print(f'    batch={b}, time={t}: mean={cross_mean:.6f}, std={cross_std:.6f}')
        if abs(cross_mean) > 0.1 or abs(cross_std - 1.0) > 0.1:
            print(f'        [WARNING] Cross-norm may not be working correctly')

# 6. Test model forward pass
print('\n[6] Testing model forward pass...')
num_stocks = max_stocks
model_instance = model.StockTransformer(
    input_dim=feature_dim,
    config=config.config,
    num_stocks=num_stocks
)
print(f'    Model created: d_model={config.config["d_model"]}, nhead={config.config["nhead"]}')

try:
    output = model_instance(sequences)
    print(f'    [OK] Forward pass output shape: {output.shape}')
except Exception as e:
    print(f'    [ERROR] Forward pass: {e}')
    import traceback; traceback.print_exc()
    raise

# 7. Test loss computation
print('\n[7] Testing loss computation...')
criterion = train.PairwiseRankingLoss(margin=config.config['margin'])
masks = result['masks']
relevance = result['relevance']

try:
    loss = criterion(output, relevance, masks)
    print(f'    [OK] Loss computed: {loss.item():.6f}')
    if torch.isnan(loss) or torch.isinf(loss):
        print(f'    [FATAL] Loss is NaN or Inf!')
        raise ValueError('Loss is NaN or Inf')
except Exception as e:
    print(f'    [ERROR] Loss computation: {e}')
    import traceback; traceback.print_exc()
    raise

print('\n' + '='*60)
print('DRY RUN PASSED - Phase1 Refactoring Successful!')
print('='*60)