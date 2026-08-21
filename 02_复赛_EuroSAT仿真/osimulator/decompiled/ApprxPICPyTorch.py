# Source Generated with Decompyle++
# File: ApprxPICPyTorch.cpython-39.pyc (Python 3.9)

import torch
from torch.nn import Module
from torch.nn import Parameter
import torch.nn.functional
F = functional
nn
import numpy as np
import entrance
import sys
import time
import math
from typing import Any, Union
from osimulator.Preprocess2 import pre_process2
import torch._dynamo as torch
torch._dynamo.config.suppress_errors = True

class ApprxPICPyTorch(Module):
    
    def __init__(self = None, device_type = None, physical_k_dimension = None, physical_n_dimension = None, n_dimension = None, k_dimension = None, input_precision = None, weight_precision = None, output_precision = None, instFolder = None, int2uint = None):
        super().__init__()
        self.device_type = device_type
        self.dims = 3
        self.input_precision = input_precision
        self.weight_precision = weight_precision
        self.output_precision = output_precision
        self.instFolder = instFolder
        self.int2uint = int2uint
        self._laser_scale = 1
        if n_dimension % physical_n_dimension != 0:
            Warning('n_dimension should be multiple of physical_n_dimension')
            sys.exit(0)
        else:
            self.repeat_n = n_dimension // physical_n_dimension
        if k_dimension % physical_k_dimension != 0:
            Warning('k_dimension should be multiple of physical_k_dimension')
            sys.exit(0)
        else:
            self.repeat_k = 1
        self.physical_k_dimension = physical_k_dimension
        self.k_dimension = k_dimension
        self.physical_n_dimension = physical_n_dimension
        self.n_dimension = n_dimension
        noise_mse_tmp = entrance.load_instance_pickle(f'''{instFolder}/noise_mse.inst''', None)
        gain_tmp = entrance.load_instance_pickle(f'''{instFolder}/gain.inst''', None)
        real_lsb_tmp = entrance.load_instance_pickle(f'''{instFolder}/real_lsb.inst''', None)
        offset_error_tmp = entrance.load_instance_pickle(f'''{instFolder}/offset_error.inst''', None)
        offset_tmp = entrance.load_instance_pickle(f'''{instFolder}/offset.inst''', None)
        scaling_factor_tmp = entrance.load_instance_pickle(f'''{instFolder}/scaling_factor.inst''', None)
        self.noise_mse = Parameter(torch.tensor(noise_mse_tmp, torch.float32, **('dtype',)).repeat(self.repeat_n), False, **('requires_grad',)).to(device_type)
        self.gain = Parameter(torch.tensor(gain_tmp, torch.float32, **('dtype',)).repeat(self.repeat_n), False, **('requires_grad',)).to(device_type)
        self.real_lsb = Parameter(torch.tensor(real_lsb_tmp, torch.float32, **('dtype',)).repeat(self.repeat_n), False, **('requires_grad',)).to(device_type)
        self.offset_error = Parameter(torch.tensor(offset_error_tmp, torch.float32, **('dtype',)).repeat(self.repeat_n), False, **('requires_grad',)).to(device_type)
        self.offset = Parameter(torch.tensor(offset_tmp, torch.float32, **('dtype',)).repeat(self.repeat_n), False, **('requires_grad',)).to(device_type)
        self.scaling_factor = Parameter(torch.tensor(scaling_factor_tmp, torch.float32, **('dtype',)).repeat(self.repeat_n), False, **('requires_grad',)).to(device_type)
        (self.mapping_factor_uint2bit, self.lut_sum_uint2_) = self._get_lut('uint2', instFolder, **('inputType', 'instFolder'))
        (self.mapping_factor_int2bit, self.lut_sum_int2_) = self._get_lut('int2', instFolder, **('inputType', 'instFolder'))
        (self.mapping_factor_uint4bit, self.lut_sum_uint4_) = self._get_lut('uint4', instFolder, **('inputType', 'instFolder'))
        (self.mapping_factor_int4bit, self.lut_sum_int4_) = self._get_lut('int4', instFolder, **('inputType', 'instFolder'))
        (self.mapping_factor_uint8bit, self.lut_sum_uint8_) = self._get_lut('uint8', instFolder, **('inputType', 'instFolder'))
        (self.mapping_factor_int8bit, self.lut_sum_int8_) = self._get_lut('int8', instFolder, **('inputType', 'instFolder'))
        self.output_scope = 2 ** self.output_precision - 1

    
    def _index_calc(self, input_mapping_factor, weight_mapping_factor, weight_precision, input_bitwidth, weight_bitwidth):
        return (lambda .0 = None: [ int(i * input_mapping_factor * 2 ** weight_precision + j * weight_mapping_factor) for i in .0 for j in range(2 ** weight_bitwidth) ])(range(2 ** input_bitwidth))

    
    def _uint_mapping_factor_calc(self, bitwidth, inputbit):
        return math.floor((2 ** bitwidth - 1) / (2 ** inputbit - 1))

    
    def _int_mapping_factor_calc(self, bitwidth, inputbit):
        return min(math.floor(2 ** (bitwidth - 1) / 2 ** (inputbit - 1)), math.floor((2 ** (bitwidth - 1) - 1) / (2 ** (inputbit - 1) - 1)))

    
    def _get_lut(self = None, inputType = None, instFolder = None):
        scale_params = {
            'uint4': (4, 4, self._uint_mapping_factor_calc),
            'uint8': (8, 8, self._uint_mapping_factor_calc),
            'uint2': (2, 2, self._uint_mapping_factor_calc),
            'int4': (4, 4, self._int_mapping_factor_calc),
            'int8': (8, 8, self._int_mapping_factor_calc),
            'int2': (2, 2, self._int_mapping_factor_calc) }
        input_bitwidth = scale_params[inputType][0]
        weight_bitwidth = scale_params[inputType][1]
        input_mapping_factor = scale_params[inputType][2](self.input_precision, input_bitwidth)
        wght_mapping_factor = self._int_mapping_factor_calc(self.weight_precision, weight_bitwidth)
        print('wght_mapping_factor: ', wght_mapping_factor)
        print('input_mapping_factor: ', input_mapping_factor)
        idx = self._index_calc(input_mapping_factor, wght_mapping_factor, self.weight_precision, input_bitwidth, weight_bitwidth)
        mapping_sum0_tmp = entrance.load_instance_pickle(f'''{instFolder}/mapping_sum0.inst''', None)
        mapping_sum1_tmp = entrance.load_instance_pickle(f'''{instFolder}/mapping_sum1.inst''', None)
        mapping_sum0 = mapping_sum0_tmp.tolist()
        mapping_sum1 = mapping_sum1_tmp.tolist()
        lut_sum0 = torch.tensor(mapping_sum0, torch.float32, **('dtype',)).reshape((1, self.physical_n_dimension, self.physical_k_dimension, -1)).contiguous()[(:, :, :, idx)].repeat((1, self.repeat_n, self.repeat_k, 1)).transpose(2, 1)
        lut_sum1 = torch.tensor(mapping_sum1, torch.float32, **('dtype',)).reshape((1, self.physical_n_dimension, self.physical_k_dimension, -1)).contiguous()[(:, :, :, idx)].repeat((1, self.repeat_n, self.repeat_k, 1)).transpose(2, 1)
        lut_sum = Parameter(torch.stack((lut_sum0, lut_sum1)).unsqueeze(0).expand(2, -1, -1, -1, -1, -1), False, **('requires_grad',)).to(self.device_type)
        return (input_mapping_factor * wght_mapping_factor, lut_sum)

    
    def matmul_impl(self, query, input_type):
        '''
        Process the query tocalculate the noisy matmul output using the given input type.
        
        Args:
            query: The query tensor. that combined with the input and weight tensor
            input_type: The type of input, e.g., "uint4", "int4", "uint8".
            
        Returns:
            res: The resulting tensor after processing the query.
        '''
        query = query.unsqueeze(-1)
        org_shape = query.shape
        query = query.reshape((2, -1, org_shape[3], org_shape[4], org_shape[5]))
        iter = query.shape[1]
        if input_type == 'uint4':
            lutv0 = torch.gather(self.lut_sum_uint4_.expand(-1, -1, iter, -1, -1, -1), -1, query.unsqueeze(1).type(torch.long, True, **('non_blocking',)).expand(-1, 2, -1, -1, -1, -1)).sum(-3).reshape((2, 2, org_shape[1], org_shape[2], org_shape[4]))
        elif input_type == 'int4':
            lutv0 = torch.gather(self.lut_sum_int4_.expand(-1, -1, iter, -1, -1, -1), -1, query.unsqueeze(1).type(torch.long, True, **('non_blocking',)).expand(-1, 2, -1, -1, -1, -1)).sum(-3).reshape((2, 2, org_shape[1], org_shape[2], org_shape[4]))
        elif input_type == 'uint8':
            lutv0 = torch.gather(self.lut_sum_uint8_.expand(-1, -1, iter, -1, -1, -1), -1, query.unsqueeze(1).type(torch.long, True, **('non_blocking',)).expand(-1, 2, -1, -1, -1, -1)).sum(-3).reshape((2, 2, org_shape[1], org_shape[2], org_shape[4]))
        elif input_type == 'int8':
            lutv0 = torch.gather(self.lut_sum_int8_.expand(-1, -1, iter, -1, -1, -1), -1, query.unsqueeze(1).type(torch.long, True, **('non_blocking',)).expand(-1, 2, -1, -1, -1, -1)).sum(-3).reshape((2, 2, org_shape[1], org_shape[2], org_shape[4]))
        elif input_type == 'int2':
            lutv0 = torch.gather(self.lut_sum_int2_.expand(-1, -1, iter, -1, -1, -1), -1, query.unsqueeze(1).type(torch.long, True, **('non_blocking',)).expand(-1, 2, -1, -1, -1, -1)).sum(-3).reshape((2, 2, org_shape[1], org_shape[2], org_shape[4]))
        elif input_type == 'uint2':
            lutv0 = torch.gather(self.lut_sum_uint2_.expand(-1, -1, iter, -1, -1, -1), -1, query.unsqueeze(1).type(torch.long, True, **('non_blocking',)).expand(-1, 2, -1, -1, -1, -1)).sum(-3).reshape((2, 2, org_shape[1], org_shape[2], org_shape[4]))
        if self._laser_scale != 1:
            lutv0.mul_(self._laser_scale)
        lutv0.add_(torch.randn_like(lutv0) * self.noise_mse[:org_shape[4]]).mul_(self.gain[:org_shape[4]]).div_(self.real_lsb[:org_shape[4]]).sub_(self.offset_error[:org_shape[4]]).round_()
        adc_out = (lutv0[(:, 0, :, :, :)] - lutv0[(:, 1, :, :, :)]) + self.offset[:org_shape[4]]
        if adc_out.max() > self.output_scope:
            print('Warning: The result of the matrix multiplication is out of range')
        if adc_out.min() < 0:
            print('Warning: The result of the matrix multiplication is out of range')
        res = torch.clip(adc_out, 0, self.output_scope)
        return res

    
    def forward(self, x, y, inputType, seed, laser_scale, gain_scale = None, noise_scale = None, freq_scale = None, adaptive = ('uint4', None, 1, 1, 1, 1, False, 'true'), calc_type = {
        'x': torch.Tensor,
        'y': torch.Tensor,
        'inputType': str,
        'seed': Union[(int, None)],
        'laser_scale': float,
        'gain_scale': float,
        'noise_scale': float,
        'freq_scale': float,
        'adaptive': bool,
        'calc_type': str,
        'return': torch.Tensor }):
        ''': 
        Performs 3D matrix multiplication with padding and mapping factors.
        Args:
            x: The first input matrix of shape (b, m, k).
            y: The second input matrix of shape (b, k, n).
            input_type: the type of the input (e.g., "uint4", "int4", "uint8")
            seed: The random seed for reproducibility.
           Returns:
            The result of the matrix multiplication and the total number of elements in the result.
        '''
        if gain_scale != 1:
            self.gain = self.gain * gain_scale
            self.scaling_factor = self.scaling_factor / gain_scale
        if noise_scale != 1:
            self.noise_mse = self.noise_mse * noise_scale
        if freq_scale != 1:
            self.noise_mse = self.noise_mse * sqrt(freq_scale)
        if laser_scale != 1:
            self._laser_scale = laser_scale
            self.scaling_factor = self.scaling_factor / laser_scale
        k_size_in_tile = self.physical_k_dimension * self.repeat_k
        n_size_in_tile = self.physical_n_dimension * self.repeat_n
        z = None
        adaptive_scales = None
        constants = None
        (b, m, n, n_pad, n_ntile, k_ntile, adaptive_scales, constants, x_tiles, y_tiles, inputType, input1, z) = pre_process2(x, y, k_size_in_tile, n_size_in_tile, inputType, self.device_type, seed, adaptive, self.int2uint, **('device_type', 'seed', 'adaptive', 'int2uint'))
        if inputType == 'uint4' or inputType == 'int4':
            x_tiles.mul_(16)
            y_tiles.add_(8)
            input1.mul_(16)
        elif inputType == 'uint8' or inputType == 'int8':
            x_tiles.mul_(256)
            y_tiles.add_(128)
            input1.mul_(256)
        elif inputType == 'uint2' or inputType == 'int2':
            x_tiles.mul_(4)
            y_tiles.add_(2)
            input1.mul_(4)
        for i in range(n_ntile):
            z[(i, :, :, :)] = self.matmul(x_tiles, y_tiles[(i, :, :, :)], input1, inputType, seed, **('input0_tensor', 'weight_tensor', 'input1_tensoUnsupported opcode: LOAD_ASSERTION_ERROR (101)
r', 'input_type', 'seed'))
        if constants != None:
            z = z - constants
            del constants
        if adaptive:
            z = z * x_tiles.shape[2] / adaptive_scales
            del adaptive_scales
        if gain_scale != 1:
            self.gain = self.gain / gain_scale
            self.scaling_factor = self.scaling_factor * gain_scale
        if noise_scale != 1:
            self.noise_mse = self.noise_mse / noise_scale
        if freq_scale != 1:
            self.noise_mse = self.noise_mse / sqrt(freq_scale)
        if laser_scale != 1:
            self._laser_scale = 1
            self.scaling_factor = self.scaling_factor * laser_scale
        del x_tiles
        del y_tiles
        del input1
        return z.permute(1, 2, 0, 3).reshape(b, k_ntile, m, n + n_pad).sum(1, torch.int32, **('axis', 'dtype'))[(:, :, :n)]

    
    def matmul(self, input0_tensor = None, weight_tensor = None, input1_tensor = None, input_type = (None,), seed = {
        'input0_tensor': torch.Tensor,
        'weight_tensor': torch.Tensor,
        'input1_tensor': torch.Tensor,
        'input_type': str }):
        if input_type == 'uint4':
            mapping_factor = self.mapping_factor_uint4bit
        elif input_type == 'int4':
            mapping_factor = self.mapping_factor_int4bit
        elif input_type == 'uint8':
            mapping_factor = self.mapping_factor_uint8bit
        elif input_type == 'int8':
            mapping_factor = self.mapping_factor_int8bit
        elif input_type == 'uint2':
            mapping_factor = self.mapping_factor_uint2bit
        elif input_type == 'int2':
            mapping_factor = self.mapping_factor_int2bit
        input0_query = input0_tensor.unsqueeze(-1)
        weight_qeury = weight_tensor.unsqueeze(1)
        input1_query = input1_tensor.unsqueeze(-1)
        query = torch.stack((input0_query.add(weight_qeury), input1_query.add(weight_qeury))).type(torch.long, True, **('non_blocking',))
        resI = self.matmul_impl(query, input_type)
        return torch.round((resI[0] - resI[1]) * self.scaling_factor[:resI.shape[-1]] / mapping_factor)

    
    def matmul_omac(self, input_tensor = None, weight_tensor = None, input_type = None, seed = (None, None), instance = {
        'input_tensor': torch.Tensor,
        'weight_tensor': torch.Tensor,
        'input_type': str }):
        pass
    # WARNING: Decompyle incomplete

    __classcell__ = None

if __name__ == '__main__':
    start = time.time()
    import os
    DEVICE = 'cuda'
    dst_path = os.path.dirname(os.path.abspath(__file__)) + '/osimulator/models/pace2/8X8_4a4w8o_dacenob5.5_power0.01_noise2e-11_1gclock_mean-1.11_std27'
    dir_cur = os.path.dirname(os.path.abspath(__file__))
    seed = None
    b = 1
    m = 3136
    k = 756
    n = 64
    in_bit = 4
    wght_bit = 4
    input_tensors = torch.randint(0, 2 ** in_bit, (b, m, k), torch.int32, DEVICE, **('low', 'high', 'size', 'dtype', 'device'))
    zeros = torch.zeros_like(input_tensors)
    wght_tensors = torch.randint(-2 ** (wght_bit - 1), 2 ** (wght_bit - 1), (b, k, n), torch.int32, DEVICE, **('low', 'high', 'size', 'dtype', 'device'))
    compiled_model = ApprxPICPyTorch(DEVICE, 8, 8, 8, 8, 4, 4, 8, False, dst_path, **('device_type', 'physical_k_dimension', 'physical_n_dimension', 'k_dimension', 'n_dimension', 'input_precision', 'weight_precision', 'output_precision', 'int2uint', 'instFolder'))
    compiled_model = torch.compile(compiled_model)
    torch.manual_seed(55)
    end = time.time()
    print(end - start)
    for i in range(20):
        start = time.time()
        (results_model_instance_pytorch_ap_separated, _) = compiled_model(input_tensors, wght_tensors, 'uint4', True, None, **('x', 'y', 'inputType', 'adaptive', 'seed'))
        end = time.time()
        print(end - start)
    expect = torch.bmm(input_tensors.to(torch.float32), wght_tensors.to(torch.float32))
    print(f'''expect: {expect}''')
    print(f'''actual: {results_model_instance_pytorch_ap_separated}''')
    print('torch')
    entrance.dump_stats(results_model_instance_pytorch_ap_separated.cpu().numpy(), expect.cpu().numpy(), 4, 8)
