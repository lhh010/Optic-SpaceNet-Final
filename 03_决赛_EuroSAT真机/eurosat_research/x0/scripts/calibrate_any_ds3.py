# calibrate_any_ds3.py — ds3 变体参数化 per-layer 标量校准 (含 h1/h2, 同窗口)
# 由 calibrate_any.py 适配: 层链含 s1ds/s2ds (conv3s2, k=9C im2col)。
# 用真实激活分布校准 (FAKE forward 每层输入) — 贴近推理工作点。
# env: DS3_WEIGHTS_DIR, DS3_CALIB_OUT
import os, json, time, sys
import numpy as np
sys.path.insert(0, '/home/uisrc/j1')
os.environ.setdefault('DS3_WEIGHTS_DIR', '/home/uisrc/j1/weights_w075ds3')
os.environ['DS3_FAKE'] = '1'
WD = os.environ['DS3_WEIGHTS_DIR']
OUT = os.environ.get('DS3_CALIB_OUT', WD + '/../calib_ds3.json')
import run_ds3_gazelle as R

images = np.load(WD + '/test_images_j1.npy')[:64]

ws, meta = R.load_weights()
eps = float(meta.get('stem_bn_eps', 1e-5))
acts = {}
h = R.stem_forward(images, ws, meta)
acts['s1a'] = h
h = R.optical_conv1x1(h, ws['s1a'], meta['s1a_scale']); h = R.apply_bn(h, ws['s1a_bn'], eps); h = R.relu(h)
acts['s1ds'] = h
h = R.optical_conv3s2(h, ws['s1ds'], meta['s1ds_scale']); h = R.apply_bn(h, ws['s1ds_bn'], eps); h = R.relu(h)
acts['s2a'] = h
h = R.optical_conv1x1(h, ws['s2a'], meta['s2a_scale']); h = R.apply_bn(h, ws['s2a_bn'], eps); h = R.relu(h)
acts['s2b'] = h
h = R.optical_conv1x1(h, ws['s2b'], meta['s2b_scale']); h = R.apply_bn(h, ws['s2b_bn'], eps); h = R.relu(h)
acts['s2ds'] = h
h = R.optical_conv3s2(h, ws['s2ds'], meta['s2ds_scale']); h = R.apply_bn(h, ws['s2ds_bn'], eps); h = R.relu(h)
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
for name in ['s1a', 's1ds', 's2a', 's2b', 's2ds', 's3a', 's3b', 'h1', 'h2']:
    w = np.load(WD + '/%s_w.npy' % name)
    xf = acts[name]
    if name in ('s1ds', 's2ds'):
        x_flat, oh, ow = R.im2col_3x3s2(xf)
    elif xf.ndim == 4:
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
    print('%s: alpha=%.5f beta=%.1f resid_std=%.1f ideal_rms=%.0f'
          % (name, a, b, resid.std(), np.sqrt((ideal**2).mean())), flush=True)

json.dump(calib, open(OUT, 'w'), indent=2)
print('saved', OUT)
