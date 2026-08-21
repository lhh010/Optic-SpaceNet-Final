import torch
sd = torch.load('weights/j1_r3_J1_long_best.pth', map_location='cpu')
print('== 官方 J1 long 全部键 ==')
for k, v in sd.items():
    print(f'  {k}: {tuple(v.shape)}')
print('== M6 v8 权重 ==')
sd6 = torch.load('weights/m6_j1_v8probe15.pth', map_location='cpu')
print(f'  keys={len(sd6)} head keys:',
      {k: tuple(v.shape) for k, v in sd6.items() if 'head' in k})
