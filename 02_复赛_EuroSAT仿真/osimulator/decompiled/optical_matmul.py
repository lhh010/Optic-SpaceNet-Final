# Source Generated with Decompyle++
# File: optical_matmul.cpython-39.pyc (Python 3.9)

import os
import sys
test_analog_root = os.path.dirname(os.path.abspath(__file__)) + '/'
test_photonic_root = test_analog_root
sys.path.append(f'''{test_analog_root}/library''')
sys.path.append(f'''{test_analog_root}..''')
import torch
import torch.nn.functional
F = functional
nn
from torch import nn
import pdb
import time
import logging
from typing import Any, Union
import entrance
import numpy as np
from device_info import compass_forward_hook
from osimulator.PreProcess import inject_variables_into_adc

class oMAC_Matmul(torch.nn.Module):
    
    def __init__(self = None, input_mapping_factor = None, wght_mapping_factor = None):
        super(oMAC_Matmul, self).__init__()
        self.input_mapping_factor = input_mapping_factor
        self.wght_mapping_factor = wght_mapping_factor
        self.count_timer = []
        self.total_timer = 0
        self.seed_compass = None
        self.in_bit_compass = 8
        self.wght_bit_compass = 8
        self.dst_path_compass = os.path.dirname(os.path.abspath(__file__)) + '/models/gazelle/8X2_8a8w12o_dacenob7.5_power0.015_noise9e-11_10mclock_mean-1.49_std5.31/'
        self.model_instance_compass = entrance.load_model(self.dst_path_compass + f'''/compass_with_lut_seed_{self.seed_compass}.inst''')
        inject_variables_into_adc(self.model_instance_compass)
        self.in_bit_pace2 = 4
        self.wght_bit_pace2 = 4
        self.seed_pace2 = None
        self.dst_path_8_pace2 = os.path.dirname(os.path.abspath(__file__)) + '/models/pace2/8X8_4a4w8o_dacenob5.5_power0.01_noise2e-11_1gclock_mean-1.11_std17'
        self.model_instance_8_pace2 = entrance.load_model(self.dst_path_8_pace2 + f'''/compass_with_lut_seed_{self.seed_pace2}.inst''')
        inject_variables_into_adc(self.model_instance_8_pace2)
        self.dst_path_32_pace2 = os.path.dirname(os.path.abspath(__file__)) + '/models/pace2/32X2_4a4w8o_dacenob5.5_power0.02_noise2e-11_1gclock_mean-5.95_std38.3'
        self.model_instance_32_pace2 = entrance.load_model(self.dst_path_32_pace2 + f'''/compass_with_lut_seed_{self.seed_pace2}.inst''')
        inject_variables_into_adc(self.model_instance_32_pace2)

    
    def fixed_quant(self, x, device_mode):
        if device_mode == 'COMPASS':
            wght_bit = self.wght_bit_compass
        elif device_mode == 'PACE2_8' or device_mode == 'PACE2_32':
            wght_bit = self.wght_bit_pace2
        else:
            wght_bit = 32
        psize = (1 << wght_bit - 1) - 1
        scale = max(np.max(x), -np.min(x)) / psize
        xq = x / scale
        xq = np.clip(xq, -psize, psize)
        xq = np.round(xq)
        if wght_bit == 8:
            xq = xq.astype(np.int8)
        elif wght_bit == 4:
            xq = xq.astype(np.int16)
        elif wght_bit == 32:
            xq = xq.astype(np.int32)
        else:
            print('ERROR: unsupported bw=%d' % wght_bit)
            exit(-1)
        return (xq, scale, 0)

    
    def ufixed_quant(self, x, device_mode):
        if device_mode == 'COMPASS':
            in_bit = self.in_bit_compass
        elif device_mode == 'PACE2_8' or device_mode == 'PACE2_32':
            in_bit = self.in_bit_pace2
        else:
            in_bit = 32
        psize = (1 << in_bit) - 1
        scale = np.max(np.max(x) - np.min(x)) / psize
        offset = -int(round(np.min(x) / scale))
        xq = x / scale + offset
        xq = np.clip(xq, 0, psize)
        xq = np.round(xq)
        if in_bit == 8:
            xq = xq.astype(np.uint8)
        elif in_bit == 4:
            xq = xq.astype(np.uint16)
        elif in_bit == 32:
            xq = xq.astype(np.uint32)
        else:
            print('ERROR: unsupported bw=%d' % in_bit)
            exit(-1)
        return (xq, scale, offset)

    
    def dequant(self, xq, scale, offset):
        return (xq.astype(np.int32) - offset) * scale

    
    def dequant_res(self, input_xq, weights_q, results):
        (Q_x, S_x, Z_x) = input_xq
        (Q_w, S_w, Z_w) = weights_q
        res = np.matmul(self.dequant(Q_x, S_x, Z_x), self.dequant(Q_w, S_w, Z_w))
        return res

    
    def opt_matmul(self, x, y, input_mapping_factor = None, wght_mapping_factor = None, seed = None, model_instance = {
        'x': torch.tensor,
        'y': torch.tensor,
        'input_mapping_factor': float,
        'wght_mapping_factor': float,
        'seed': Union[(int, None)],
        'model_instance': entrance.Any,
        'return': torch.Tensor }):
        omac_3d = entrance.matmul_omac_3d(x.numpy(), y.numpy(), 1, 1, None, model_instance, **('x', 'y', 'input_mapping_factor', 'wght_mapping_factor', 'seed', 'model_instance'))
        return omac_3d[0]

    
    def preprocess_restrictions_tensor(self, tensor1, tensor2, device_mode):
        dim1 = tensor1.shape
        dim2 = tensor2.shape
        if device_mode == 'COMPASS' and device_mode == 'PACE2_32' or device_mode == 'PACE2_8':
            if len(dim1) == len(dim2) and len(dim1) == 2:
                tensor1 = tensor1.unsqueeze(0)
                tensor2 = tensor2.unsqueeze(0)
                new_dim1 = tensor1.shape
                new_dim2 = tensor2.shape
                if new_dim1[0] == new_dim2[0] and new_dim1[2] == new_dim2[1]:
                    return (tensor1, tensor2, True)
                if None[0] == new_dim2[0] and new_dim1[1] == new_dim2[1]:
                    return (tensor1, tensor2, False)
                raise None('RuntimeError: mat1 and mat2 shapes cannot be multiplied ({}x{} and {}x{})'.format(new_dim1[0], new_dim1[1], new_dim2[0], new_dim2[1]))
        elif len(dim1) == len(dim2) and len(dim1) == 3:
            print('**** len(dim1) == len(dim2) == 3')
            if dim1[0] == dim2[0] and dim1[2] == dim2[1]:
                return (tensor1, tensor2, True)
            if None[0] == dim2[0] and dim1[1] == dim2[1]:
                return (tensor1, tensor2, False)
            if (None[0] == dim2[0] or dim1[1] != dim2[1] or dim1[0] != dim2[0]) and dim1[1] == dim2[1]:
                raise Exception('RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [{}, {}] but got: [{}, {}].'.format(dim1[0], dim1[1], dim2[0], dim2[1]))
        else:
            raise Exception('RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0')
        return (tensor1, tensor2, False)

    
    def forward(self, input_x, weights, device_mode = ('COMPASS',)):
        '''
        input_x: 输入数据,类型为tensor or numpy
        weights: 权重数据,类型为tensor or numpy
        device_mode: 
            - COMPASS : compass硬件仿真
            - PACE2_32 : pace2硬件仿真32x32
            - PACE2_8 : pace2硬件仿真8x8
            - CPU: cpu运行
        '''
        if not torch.is_tensor(input_x):
            input_x = torch.from_numpy(input_x.astype(np.float32))
        if not torch.is_tensor(weights):
            weights = torch.from_numpy(weights.astype(np.float32))
        dim_nums = len(input_x.shape)
        (input_x, weights, omac_flag) = self.preprocess_restrictions_tensor(input_x, weights, device_mode)
        if omac_flag:
            input_x_ = input_x.numpy()
            input_xq = self.ufixed_quant(input_x_, device_mode)
            input_x = torch.from_numpy(input_xq[0].astype(np.int32))
            weights_ = weights.numpy()
            weights_q = self.fixed_quant(weights_, device_mode)
            wght_tensors = torch.from_numpy(weights_q[0].astype(np.int32))
            if device_mode == 'PACE2_32':
                model_instance = self.model_instance_32_pace2
            elif device_mode == 'PACE2_8':
                model_instance = self.model_instance_8_pace2
            else:
                model_instance = self.model_instance_compass
            results_model_instance_ap_separated = self.opt_matmul(input_x, wght_tensors, 1, 1, None, model_instance, **('x', 'y', 'input_mapping_factor', 'wght_mapping_factor', 'seed', 'model_instance'))
            if device_mode == 'COMPASS':
                (b, m, k) = input_x.shape
                n = wght_tensors.shape[2]
                compass_forward_hook(self, b, m, k, n)
            np_results_ap_separated = results_model_instance_ap_separated
            uqnp_results_ap_separated = self.dequant_res(input_xq, weights_q, np_results_ap_separated)
            results_ap_separated = torch.from_numpy(uqnp_results_ap_separated)
            results = results_ap_separated
            if dim_nums == 2:
                return results.squeeze(0, **('dim',))
            return None
        output_tensor = torch.matmul(input_x, weights)
        return output_tensor

    
    def omatmul(self, input_x, weights, device_mode = ('COMPASS',)):
        t0 = time.time()
        output = self.forward(input_x, weights, device_mode)
        t1 = time.time()
        infer_latency = round(t1 - t0, 6) * 1000
        self.count_timer.append(infer_latency)
        self.total_timer += infer_latency
        logging.info('infer times:{} ms'.format(infer_latency))
        return output

    __classcell__ = None

if __name__ == '__main__':
    np.random.seed(100)
    opt_matmul = oMAC_Matmul()
    x = np.random.uniform(-1, 1, (4, 2, 3), **('low', 'high', 'size'))
    print('x = ', x)
    weights = np.random.uniform(-1, 1, (4, 3, 2), **('low', 'high', 'size'))
    torch_res = torch.matmul(torch.from_numpy(x), torch.from_numpy(weights))
    print('torch_res = ', torch_res)
    print('torch_res shape = ', torch_res.shape)
    results = opt_matmul.omatmul(x, weights, 'PACE2_32', **('device_mode',))
    omac_res = results
    print('omac_res = ', omac_res)
    print('omac_res shape = ', omac_res.shape)
