import os, json, time
import numpy as np
from compass_sdk.fast_calibration.compass_lib import compass_matmul, compass_init
compass_init(150); time.sleep(3)

WD='/home/uisrc/j1/weights_j1'
calib = json.load(open('/home/uisrc/j1/calib_j1.json'))
meta = json.load(open(WD+'/meta.json'))

def engine(x_u8, w_i8):
    w = w_i8.astype(np.int8)
    outs = []
    for i in range(0, x_u8.shape[0], 2):
        outs.append(compass_matmul(x_u8[i:i+2], w).astype(np.float64))
    return np.vstack(outs)

for name in ['s1a','s2a','s2b','s3a','s3b','h1','h2']:
    w = np.load(WD+f'/{name}_w.npy')  # (C_out, C)
    k, n = w.shape[1], w.shape[0]
    rng = np.random.RandomState(hash(name)%2**32)
    xs = []
    for _ in range(8):
        xs.append(rng.randint(0,256,size=(2,k)).astype(np.uint8))
    X = np.vstack(xs).astype(np.float64)
    ideal = X @ w.T.astype(np.float64)
    hw = engine(X.astype(np.uint8), w.astype(np.int8).T)
    a,b = calib[name]['alpha'], calib[name]['beta']
    corr_after = (hw-b)/a
    c = np.corrcoef(ideal.ravel(), corr_after.ravel())[0,1]
    rel = np.abs(ideal-corr_after).mean()/np.abs(ideal).mean()
    print(f'{name}: post-calib corr={c:.5f} rel_mae={rel*100:.2f}%', flush=True)
