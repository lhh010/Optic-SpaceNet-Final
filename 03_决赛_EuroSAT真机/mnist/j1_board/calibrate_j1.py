# 对 J1 每个光计算层校准 alpha/beta: hw = alpha*ideal + beta
import os, json, time
import numpy as np
from compass_sdk.fast_calibration.compass_lib import compass_matmul, compass_init
compass_init(150); time.sleep(3)

WD='/home/uisrc/j1/weights_j1'
meta = json.load(open(WD+'/meta.json'))

def engine(x_u8, w_i8, tile=2):
    w = w_i8.astype(np.int8)
    outs = []
    for i in range(0, x_u8.shape[0], tile):
        outs.append(compass_matmul(x_u8[i:i+tile], w).astype(np.float64))
    return np.vstack(outs)

def fit_layer(name, w_int, n_cal=32):
    k, n = w_int.shape  # w_int: (k, n) = (C_in, C_out)
    rng = np.random.RandomState(7)
    xs = []
    for _ in range(n_cal):
        # 激活模拟: uint8 全范围, 模拟真实分布
        xs.append(rng.randint(0, 256, size=(2, k)).astype(np.uint8))
    X = np.vstack(xs).astype(np.float64)
    ideal = X @ w_int.astype(np.float64)
    hw = engine(X.astype(np.uint8), w_int.T.astype(np.int8))  # engine 内部转置?
    return X, ideal, hw

# 直接按 optical_conv1x1 布局: w_int (C_out, C), x (m, C) -> 用 w_int.T (C, C_out)
layers = ['s1a','s2a','s2b','s3a','s3b','h1','h2']
calib = {}
for name in layers:
    w = np.load(WD+f'/{name}_w.npy')          # (C_out, C)
    k, n = w.shape[1], w.shape[0]
    rng = np.random.RandomState(hash(name) % 2**32)
    xs = []
    for _ in range(16):
        xs.append(rng.randint(0, 256, size=(2, k)).astype(np.uint8))
    X = np.vstack(xs).astype(np.float64)      # (32, C)
    ideal = X @ w.T.astype(np.float64)        # (32, C_out)
    hw = engine(X.astype(np.uint8), w.astype(np.int8).T)  # (C, C_out)
    a, b = np.polyfit(ideal.ravel(), hw.ravel(), 1)
    resid = hw.ravel() - (a*ideal.ravel()+b)
    calib[name] = {'alpha': float(a), 'beta': float(b), 'resid_std': float(resid.std())}
    print(f'{name} ({k}x{n}): alpha={a:.5f} beta={b:.1f} resid_std={resid.std():.1f} ideal_rms={np.sqrt((ideal**2).mean()):.0f}', flush=True)

json.dump(calib, open('/home/uisrc/j1/calib_j1.json','w'), indent=2)
print('saved calib_j1.json')
