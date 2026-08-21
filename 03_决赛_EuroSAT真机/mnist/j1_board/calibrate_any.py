# calibrate_any.py — 参数化 per-layer 校准 (env: J1_WEIGHTS_DIR, J1_CALIB_OUT)
# 用真实激活分布校准 (FAKE forward 每层输入) — 贴近推理工作点
import os, json, time, sys
import numpy as np
sys.path.insert(0, '/home/uisrc/j1')
os.environ.setdefault('J1_WEIGHTS_DIR', '/home/uisrc/j1/weights_j1')
os.environ['J1_FAKE'] = '1'
WD = os.environ['J1_WEIGHTS_DIR']
OUT = os.environ.get('J1_CALIB_OUT', WD + '/../calib_any.json')
import run_j1_gazelle as R

images = np.load(WD + '/test_images_j1.npy')[:64]

ws, meta = R.load_weights()
eps = float(meta.get('stem_bn_eps', 1e-5))
acts = {}
h = R.stem_forward(images, ws, meta)
acts['s1a'] = h
h = R.optical_conv1x1(h, ws['s1a'], meta['s1a_scale']); h = R.apply_bn(h, ws['s1a_bn'], eps); h = R.relu(h); h = R.pool2d(h, 2)
acts['s2a'] = h
h = R.optical_conv1x1(h, ws['s2a'], meta['s2a_scale']); h = R.apply_bn(h, ws['s2a_bn'], eps); h = R.relu(h)
acts['s2b'] = h
h = R.optical_conv1x1(h, ws['s2b'], meta['s2b_scale']); h = R.apply_bn(h, ws['s2b_bn'], eps); h = R.relu(h); h = R.pool2d(h, 2)
acts['s3a'] = h
h = R.optical_conv1x1(h, ws['s3a'], meta['s3a_scale']); h = R.apply_bn(h, ws['s3a_bn'], eps); h = R.relu(h)
acts['s3b'] = h
h = R.optical_conv1x1(h, ws['s3b'], meta['s3b_scale']); h = R.apply_bn(h, ws['s3b_bn'], eps); h = R.relu(h)
g = h.mean(axis=(2, 3)); acts['h1'] = g
z = R.optical_fc(g, ws['h1'], meta['h1_scale']); z = R.relu(z); acts['h2'] = z

from compass_sdk.fast_calibration.compass_lib import compass_matmul, compass_init
compass_init(150); time.sleep(3)

def engine(x_u8, w_i8):
    w = w_i8.astype(np.int8)
    outs = []
    for i in range(0, x_u8.shape[0], 2):
        outs.append(compass_matmul(x_u8[i:i+2], w).astype(np.float64))
    return np.vstack(outs)

calib = {}
for name in ['s1a', 's2a', 's2b', 's3a', 's3b', 'h1', 'h2']:
    w = np.load(WD + f'/{name}_w.npy')
    xf = acts[name]
    if xf.ndim == 4:
        B, C, H, W = xf.shape
        x_flat = xf.transpose(0, 2, 3, 1).reshape(B * H * W, C)
    else:
        x_flat = xf
    x_int, x_scale, x_zp = R.quantize_act(x_flat)
    ideal = x_int.astype(np.float64) @ w.T.astype(np.float64)
    hw = engine(x_int.astype(np.uint8), w.astype(np.int8).T)
    a, b = np.polyfit(ideal.ravel(), hw.ravel(), 1)
    resid = hw.ravel() - (a * ideal.ravel() + b)
    calib[name] = {'alpha': float(a), 'beta': float(b), 'resid_std': float(resid.std())}
    print(f'{name}: alpha={a:.5f} beta={b:.1f} resid_std={resid.std():.1f} ideal_rms={np.sqrt((ideal**2).mean()):.0f}', flush=True)

json.dump(calib, open(OUT, 'w'), indent=2)
print('saved', OUT)
