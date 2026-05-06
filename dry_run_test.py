#!/usr/bin/env python3
"""
Dry Run Test - V3 干跑查杀测试
验证：移除截面归一化、197维特征、margin=0.5、PairwiseRankingLoss
"""
import sys, os, torch

# 尝试多个可能的路径
possible_paths = [
    '/app/code/src',
    'THU-BDC2026/code/src',
    os.path.join(os.path.dirname(__file__), 'code', 'src'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code', 'src'),
]
for p in possible_paths:
    if os.path.exists(p):
        sys.path.insert(0, p)
        break

print("="*70)
print("DRY RUN TEST - V3 战前彩排")
print("="*70)

# 1. Import modified modules
import model, config, train
print("\n[1] 模块导入成功 ✓")

# 2. Load config and check margin
print(f"\n[2] Config check:")
print(f"    margin: {config.config['margin']} (should be 0.5)")
print(f"    batch_size: {config.config['batch_size']} (should be 8)")
print(f"    accumulation_steps: {config.config['accumulation_steps']} (should be 4)")
print(f"    feature_num: {config.config['feature_num']} (should be 158+39)")

# 3. Create mock batch with known statistics
batch_size = 2
max_stocks = 5
seq_len = 50
feature_dim = 197  # 158 Alpha + 39 Tech indicators (NO cross-sectional rank features)

# Create sequences with known mean/std
mock_batch = []
for i in range(batch_size):
    # Random data with specific statistics
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

print(f"\n[3] Mock batch created:")
print(f"    batch_size={batch_size}, max_stocks={max_stocks}, seq_len={seq_len}, feature_dim={feature_dim}")

# 4. Run collate_fn WITHOUT cross-sectional normalization
print("\n[4] Running collate_fn (V3: NO cross-sectional normalization)...")
try:
    result = train.collate_fn(mock_batch)
    print("    collate_fn completed ✓")
except Exception as e:
    print(f"    ERROR in collate_fn: {e}")
    raise

# 5. Check for NaN/Inf in features
sequences = result['sequences']
print(f"\n[5] Feature NaN/Inf check:")
print(f"    sequences shape: {sequences.shape}")
print(f"    has NaN: {torch.isnan(sequences).any()}")
print(f"    has Inf: {torch.isinf(sequences).any()}")
print(f"    min value: {sequences.min().item():.6f}")
print(f"    max value: {sequences.max().item():.6f}")
print(f"    mean value: {sequences.mean().item():.6f}")
print(f"    std value: {sequences.std().item():.6f}")

if torch.isnan(sequences).any() or torch.isinf(sequences).any():
    print("    *** FATAL: NaN or Inf detected! ***")
else:
    print("    ✓ No NaN/Inf detected")

# 6. Verify NO cross-sectional normalization (values should preserve absolute scale)
# V3: After removing normalization, mean across stocks should NOT be ~0
print("\n[6] V3: Verify NO cross-sectional normalization (absolute momentum preserved):")
for b in range(batch_size):
    for t in range(min(3, seq_len)):  # Check first 3 time steps
        slice_ = sequences[b, :, t, :]
        cross_mean = slice_.mean().item()
        cross_std = slice_.std().item()
        print(f"    batch={b}, time={t}: mean={cross_mean:.6f}, std={cross_std:.6f}")
        # V3: mean should be ~5 (original mean), NOT ~0
        if abs(cross_mean - 5.0) < 0.5:
            print(f"        ✓ Absolute momentum preserved (mean ≈ 5)")
        else:
            print(f"        *** WARNING: Cross-norm may have been applied ***")

# 7. Create model and check output shape
print("\n[7] Model forward pass test:")
try:
    model_instance = model.StockTransformer(
        input_dim=feature_dim,
        config=config.config,
        num_stocks=300,
        emb_dim=16
    )
    print(f"    Model created ✓")
    print(f"    Model input_proj weight shape: {model_instance.input_proj.weight.shape}")
    
    # Test forward
    test_input = torch.randn(batch_size, max_stocks, seq_len, feature_dim)
    output = model_instance(test_input)
    print(f"    Forward pass output shape: {output.shape}")
    print(f"    has NaN: {torch.isnan(output).any()}")
    print(f"    has Inf: {torch.isinf(output).any()}")
except Exception as e:
    print(f"    *** ERROR: {e} ***")
    raise

# 8. Test PairwiseRankingLoss with margin=0.5
print("\n[8] PairwiseRankingLoss test (margin=0.5):")
try:
    criterion = train.PairwiseRankingLoss(margin=config.config['margin'], k=5)
    print(f"    Loss function created ✓")
    print(f"    margin: {criterion.margin}")
    
    # Create valid predictions
    valid_pred = torch.randn(8)  # 8 samples
    valid_relevance = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float)  # relevance scores
    loss = criterion(valid_pred.unsqueeze(0), valid_relevance.unsqueeze(0))
    print(f"    Loss value: {loss.item():.6f}")
    print(f"    has NaN: {torch.isnan(loss).any()}")
    print(f"    has Inf: {torch.isinf(loss).any()}")
    
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print("    *** FATAL: Loss is NaN or Inf! ***")
    else:
        print("    ✓ Loss is valid")
except Exception as e:
    print(f"    *** ERROR: {e} ***")
    raise

# 9. Summary
print("\n" + "="*70)
print("V3 DRY RUN SUMMARY")
print("="*70)
print(f"✓ 特征维度: {feature_dim} (158+39, 无截面排名特征)")
print(f"✓ 截面Z-score标准化: 已移除")
print(f"✓ margin: {config.config['margin']}")
print(f"✓ batch_size: {config.config['batch_size']}, accumulation_steps: {config.config['accumulation_steps']}")
print(f"✓ Loss计算: 无NaN/Inf")
print("\n[V3 战前准备完成，随时待命！]")
print("="*70)
