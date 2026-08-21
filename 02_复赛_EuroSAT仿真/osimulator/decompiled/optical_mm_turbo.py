# Source Generated with Decompyle++
# File: optical_mm_turbo.cpython-39.pyc (Python 3.9)

import os
import sys
test_analog_root = os.path.dirname(os.path.abspath(__file__)) + '/'
test_photonic_root = test_analog_root
sys.path.append(f'''{test_analog_root}/library''')
sys.path.append(f'''{test_analog_root}..''')
import torch
import time
import logging
from typing import Any, Union
from osimulator.api import load_approach_model
import numpy as np
gpu_arch_type = os.getenv('GPU_ARCH_TYPE', 'sm80')

class oMAC_mm_Turbo:
    
    def __init__(self):
        osim_path = os.path.dirname(os.path.abspath(__file__))
        odk_path = os.path.join(osim_path, 'models', 'pace3', '32X16X4_4a4w10o_dacenob6_power0.32_noise1e-11_1gclock_asym_mean0.15_std18')
        compiled_model = load_approach_model(gpu_arch_type, f'''{odk_path}''', [], **('instances_path', 'gpu_devices'))
        self.matmul = compiled_model


if __name__ == '__main__':
    device = 'cuda'
    b = 1000
    m = 1
    k = 16
    n = 16
    in_bit = 4
    wght_bit = 4
    out_bit = 8
    input_type = 'int4'
    wght_type = 'int4'
    np.random.seed(100)
    opt_mm = oMAC_mm_Turbo()
    device = 'cuda:0'
    input_tensors = torch.randint(-2 ** (in_bit - 1), 2 ** (in_bit - 1), (b, m, k), torch.int32, device, **('low', 'high', 'size', 'dtype', 'device'))
    weight_tensors = torch.randint(-2 ** (wght_bit - 1), 2 ** (wght_bit - 1), (b, k, n), torch.int32, device, **('low', 'high', 'size', 'dtype', 'device'))
    result = opt_mm(input_tensors, weight_tensors, None, 'int4', **('seed', 'inputType'))
    print(result.cpu().numpy())
