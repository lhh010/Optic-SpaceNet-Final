# Source Generated with Decompyle++
# File: ApprxPICPyTorchCuda.cpython-39.pyc (Python 3.9)

import torch
from torch.nn import Module
from torch.nn import Parameter
from torch.nn import nn
import torch.nn.functional
F = functional
nn
from typing import Any, Union
import omatmul
import entrance
import os
major_version = int(torch.__version__.split('.')[0])
if major_version > 1:
    import torch._dynamo as torch
    torch._dynamo.config.suppress_errors = True

class ApprxPICPyTorchCuda(Module):
    '''
    Parameters:
        device_type (str): The type of device to use, default is "cuda".
        physical_k_dimension (int): The physical k dimension, default is 16.
        physical_n_dimension (int): The physical n dimension, default is 16.
        n_dimension (int): The n dimension, default is 256.
        k_dimension (int): The k dimension, default is 256.
        input_precision (torch.int32): The precision of the input, default is 4.
        weight_precision (torch.int32): The precision of the weight, default is 4.
        output_precision (torch.int32): The precision of the output, default is 8.
        instFolder (str): The folder to load instances from, default is an empty string.
        cuda_function (function): The cuda function to use, default is None.
    Returns:
        None
    '''
    
    def __init__(self = None, device_type = None, physical_k_dimension = None, physical_n_dimension = None, n_dimension = None, k_dimension = None, input_precision = None, weight_precision = None, output_precision = None, instFolder = None, int2uint = None, cuda_function = ('cuda', 16, 16, 256, 256, 4, 4, 8, '', False, None)):
        super(ApprxPICPyTorchCuda, self).__init__()
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
        elif 'cuda' in device_type:
            self.repeat_n = n_dimension // physical_n_dimension
        else:
            self.repeat_n = 1
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
        self.cuda_function = cuda_function
        self.noise_mse = Parameter(torch.tensor(noise_mse_tmp, torch.float32, **('dtype',)).repeat(self.repeat_n).to(device_type), False, **('requires_grad',))
        self.gain = Parameter(torch.tensor(gain_tmp, torch.float32, **('dtype',)).repeat(self.repeat_n).to(device_type), False, **('requires_grad',))
        self.real_lsb = Parameter(torch.tensor(real_lsb_tmp, torch.float32, **('dtype',)).repeat(self.repeat_n).to(device_type), False, **('requires_grad',))
        self.offset_error = Parameter(torch.tensor(offset_error_tmp, torch.float32, **('dtype',)).repeat(self.repeat_n).to(device_type), False, **('requires_grad',))
        self.offset = Parameter(torch.tensor(offset_tmp, torch.float32, **('dtype',)).repeat(self.repeat_n).to(device_type), False, **('requires_grad',))
        self.scaling_factor = Parameter(torch.tensor(scaling_factor_tmp, torch.float32, **('dtype',)).repeat(self.repeat_n).to(device_type), False, **('requires_grad',))
        (self.mapping_factor_uint2bit, self.lut_sum0_uint2_, self.lut_sum1_uint2_) = self._get_lut('uint2', instFolder, **('inputType', 'instFolder'))
        (self.mapping_factor_int2bit, self.lut_sum0_int2_, self.lut_sum1_int2_) = self._get_lut('int2', instFolder, **('inputType', 'instFolder'))
        (self.mapping_factor_uint4bit, self.lut_sum0_uint4_, self.lut_sum1_uint4_) = self._get_lut('uint4', instFolder, **('inputType', 'instFolder'))
        (self.mapping_factor_int4bit, self.lut_sum0_int4_, self.lut_sum1_int4_) = self._get_lut('int4', instFolder, **('inputType', 'instFolder'))
        (self.mapping_factor_uint8bit, self.lut_sum0_uint8_, self.lut_sum1_uint8_) = self._get_lut('uint8', instFolder, **('inputType', 'instFolder'))
        (self.mapping_factor_int8bit, self.lut_sum0_int8_, self.lut_sum1_int8_) = self._get_lut('int8', instFolder, **('inputType', 'instFolder'))
        self.calc_attribute_dicts = {
            'uint2': (16, 2, self.mapping_factor_uint2bit),
            'int2': (16, 2, self.mapping_factor_int2bit),
            'uint4': (256, 4, self.mapping_factor_uint4bit),
            'int4': (256, 4, self.mapping_factor_int4bit),
            'uint8': (65536, 8, self.mapping_factor_uint8bit),
            'int8': (65536, 8, self.mapping_factor_int8bit) }
        self.output_scope = 2 ** self.output_precision - 1

    
    def _index_calc(self, input_mapping_factor, weight_mapping_factor, weight_precision, input_bitwidth, weight_bitwidth):
        return (lambda .0 = None: [ int(i * input_mapping_factor * 2 ** weight_precision + j * weight_mapping_factor) for i in .0 for j in range(2 ** weight_bitwidth) ])(range(2 ** input_bitwidth))

    
    def _uint_mapping_factor_calc(self = None, bitwidth = None, inputbit = None):
        return entrance.floor((2 ** bitwidth - 1) / (2 ** inputbit - 1))

    
    def _int_mapping_factor_calc(self, bitwidth, inputbit):
        return min(entrance.floor(2 ** (bitwidth - 1) / 2 ** (inputbit - 1)), entrance.floor((2 ** (bitwidth - 1) - 1) / (2 ** (inputbit - 1) - 1)))

    
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
        idx = self._index_calc(input_mapping_factor, wght_mapping_factor, self.weight_precision, input_bitwidth, weight_bitwidth)
        mapping_sum0_tmp = entrance.load_instance_pickle(f'''{instFolder}/mapping_sum0.inst''', None)
        mapping_sum1_tmp = entrance.load_instance_pickle(f'''{instFolder}/mapping_sum1.inst''', None)
        mapping_sum0 = mapping_sum0_tmp.tolist()
        mapping_sum1 = mapping_sum1_tmp.tolist()
        lut_sum0 = Parameter(torch.tensor(mapping_sum0, torch.float32, **('dtype',)).reshape((self.physical_n_dimension, self.physical_k_dimension, -1)).contiguous()[(:, :, idx)].repeat((self.repeat_n, self.repeat_k, 1)).to(self.device_type), False, **('requires_grad',))
        lut_sum1 = Parameter(torch.tensor(mapping_sum1, torch.float32, **('dtype',)).reshape((self.physical_n_dimension, self.physical_k_dimension, -1)).contiguous()[(:, :, idx)].repeat((self.repeat_n, self.repeat_k, 1)).to(self.device_type), False, **('requires_grad',))
        return (input_mapping_factor * wght_mapping_factor, lut_sum0, lut_sum1)

    
    def calc(self, input, zeros, weight, Unsupported opcode: LOAD_ASSERTION_ERROR (101)
output, params, lut_sum0, lut_sum1 = None, batch = None, m = torch.jit.ignore, n = {
        'input': torch.Tensor,
        'zeros': torch.Tensor,
        'weight': torch.Tensor,
        'output': torch.Tensor,
        'params': torch.Tensor,
        'lut_sum0': torch.nn.Parameter,
        'lut_sum1': torch.nn.Parameter,
        'batch': int,
        'm': int,
        'n': int }):
        pass
    # WARNING: Decompyle incomplete

    calc = None(calc)
    
    def forward(self, x, y, inputType, laser_scale, gain_scale = None, noise_scale = None, freq_scale = None, adaptive = ('uint4', 1, 1, 1, 1, False, None), seed = {
        'x': torch.Tensor,
        'y': torch.Tensor,
        'inputType': str,
        'laser_scale': float,
        'gain_scale': float,
        'noise_scale': float,
        'freq_scale': float,
        'adaptive': bool,
        'seed': Union[(int, None)],
        'return': torch.Tensor }):
        '''
        Performs 3D matrix multiplication with padding and mapping factors.
        Args:
            x: The first input matrix of shape (b, m, k).
            y: The second input matrix of shape (b, k, n).
            input_type: the type of the input (e.g., "uint4", "int4", "uint8")
            seed: The random seed for reproducibility.
            laser_scale (float, optional): A scaling factor applied to the laser power, Defaults to 1.0.
            gain_scale (float, optional): A scaling factor applied to the gain of TIA. Defaults to 1.0.
            noise_scale (float, optional): A scaling factor applied to the noise level added to TIA. Defaults to 1.0.
            freq_scale (float, optional): A scaling factor applied to the frequency of clock. Defaults to 1.0.
            adaptive (bool, optional): A flag indicating whether adaptive computations should be used, which might include adjustments based on input characteristics. Defaults to False.
           Returns:
            The result of the matrix multiplication and the total number of elements in the result.
        '''
        torch.cuda.set_device(x.device.index)
        if gain_scale != 1:
            self.gain.data *= gain_scale
            self.scaling_factor.data /= gain_scale
        if noise_scale != 1:
            self.noise_mse.data *= noise_scale
        if freq_scale != 1:
            self.noise_mse.data *= entrance.sqrt(freq_scale)
        if laser_scale != 1:
            self._laser_scale = laser_scale
            self.scaling_factor.data /= laser_scale
        (b, m, k) = x.shape
        (b1, k1, n) = y.shape
        if b != b1 or k != k1:
            raise ValueError('Warning: Shapes of x and y do not match')
        k_size_in_tile = self.physical_k_dimension * self.repeat_k
        n_size_in_tile = self.physical_n_dimension * self.repeat_n
        k_pad = (k_size_in_tile - k % k_size_in_tile) % k_size_in_tile
        n_pad = (n_size_in_tile - n % n_size_in_tile) % n_size_in_tile
        (B, M, K, N) = (x_padded.shape[0], x_padded.shape[1], x_padded.shape[2], y_padded.shape[1])
        zeros_3d = torch.zeros_like(x_padded, torch.int32, x.device, **('dtype', 'device'))
        output = torch.empty((B, M, N), torch.int32, x.device, **('dtype', 'device'))
        (L, weight_bit, mapping_factor) = self.calc_attribute_dicts[inputType]
        if inputType == 'uint4':
            lut_sum0 = self.lut_sum0_uint4_
            lut_sum1 = self.lut_sum1_uint4_
        elif inputType == 'int4':
            lut_sum0 = self.lut_sum0_int4_
            lut_sum1 = self.lut_sum1_int4_
        elif inputType == 'uint8':
            lut_sum0 = self.lut_sum0_uint8_
            lut_sum1 = self.lut_sum1_uint8_
        elif inputType == 'int8':
            lut_sum0 = self.lut_sum0_int8_
            lut_sum1 = self.lut_sum1_int8_
        elif inputType == 'uint2':
            lut_sum0 = self.lut_sum0_uint2_
            lut_sum1 = self.lut_sum1_uint2_
        elif inputType == 'int2':
            lut_sum0 = self.lut_sum0_int2_
            lut_sum1 = self.lut_sum1_int2_
        else:
            lut_sum0 = self.lut_sum0_uint4_
            lut_sum1 = self.luUnsupported opcode: LOAD_ASSERTION_ERROR (101)
t_sum1_uint4_
        out_min = 0
        out_max = (1 << self.output_precision) - 1
        if seed == None:
            seed = entrance.time.time_ns() % 6000
        params = torch.tensor([
            float(B),
            float(M),
            float(K),
            float(N),
            float(L),
            float(n_size_in_tile),
            float(k_size_in_tile),
            float(weight_bit),
            float(out_min),
            float(out_max),
            float(mapping_factor),
            float(adaptive),
            self._laser_scale,
            float(seed)], torch.float32, x.device, **('dtype', 'device'))
        if gain_scale != 1:
            self.gain.data /= gain_scale
            self.scaling_factor.data *= gain_scale
        if noise_scale != 1:
            self.noise_mse.data /= noise_scale
        if freq_scale != 1:
            self.noise_mse.data /= entrance.sqrt(freq_scale)
        if laser_scale != 1:
            self._laser_scale = 1
            self.scaling_factor.data *= laser_scale
        if k_pad > 0:
            del x_padded
        del y_padded
        del zeros_3d
        del params
        del lut_sum0
        del lut_sum1
        return output[(:, :, :n)]

    
    def matmul_omac(self, input_tensor = None, weight_tensor = None, input_type = None, seed = (None, None), instance = {
        'input_tensor': torch.Tensor,
        'weight_tensor': torch.Tensor,
        'input_type': str }):
        pass
    # WARNING: Decompyle incomplete

    __classcell__ = None

