#!/usr/bin/env python3
import torch
# Check Pairwise_Golden model dim
ckpt = torch.load('/app/model/Pairwise_Golden/best_model.pth', map_location='cpu')
print('Pairwise_Golden input_proj.weight:', ckpt['input_proj.weight'].shape)
# Check 50_158+39 model dim
ckpt2 = torch.load('/app/model/50_158+39/best_model.pth', map_location='cpu')
print('50_158+39 input_proj.weight:', ckpt2['input_proj.weight'].shape)
# Check if they have scaler
import os
print('Pairwise_Golden scaler exists:', os.path.exists('/app/model/Pairwise_Golden/scaler.pkl'))
print('50_158+39 scaler exists:', os.path.exists('/app/model/50_158+39/scaler.pkl'))