Unsupported opcode: LOAD_ASSERTION_ERROR (101)
Unsupported opcode: LOAD_ASSERTION_ERROR (101)
# Source Generated with Decompyle++
# File: api.cpython-39.pyc (Python 3.9)

import os
import torch
import torch.nn.functional
F = functional
nn
import entrance
from osimulator import PostProcess
if torch.cuda.is_available():
    from osimulator.ApprxPICPyTorchCuda import ApprxPICPyTorchCuda
import ctypes
osim_path = os.path.dirname(os.path.abspath(__file__))
gpu_device_type_dicts = {
    'sm80': [
        'A100',
        'A800'],
    'sm86': [
        '3080',
        '3090',
        'A2000',
        'A3000',
        'A4000',
        'A5000',
        'A6000',
        'A40',
        '3060',
        '3070',
        '3050',
        'A10',
        'A16',
        'A40'],
    'sm89': [
        '4090',
        '4080',
        '6000',
        'L20',
        'L40',
        'L4'],
    'sm90': [
        'H100',
        'H200',
        'H20'],
    'gpu': [
        ''] }

def load_approach_model(device_type, instances_path, gpu_devices = None, k_dimension = None, n_dimension = None, cls_cuda = ([], 0, 0, None, False), compile_torch = {
    'device_type': str,
    'instances_path': str,
    'gpu_devices': list,
    'k_dimension': int,
    'n_dimension': int }):
    """
    Loads a model based on the device type.

    Args:
    - device_type (str): Device type; can be 'cpu' or GPU models such as 'sm80', 'sm86', 'sm89', and 'sm90'.
    - instances_path (str): File path of the model instance.
    - gpu_devices (list, optional): List of available GPU devices by storing GPU ID numbers. Defaults to an empty list (i.e., using GPU 0). If specified (e.g., [0,1,2,3]), it indicates running on multiple GPUs (GPU 0/1/2/3).
    - k_dimension (int, optional): Size of the k-dimension after expansion by the computation engine. Default value is 0, meaning  no expansion is performed and the initial k-size built into the model is used. If non-zero, it represents the expanded k-dimension size, which must be an integer multiple of the initial tile size.
    - n_dimension (int, optional): Refers to the size of the n-dimension after expansion by the computation engine. The default value is 0, which means no expansion is performed and the initial n-size built into the model is used. If the value is non-zero, it represents the expanded n-dimension size, which must be an integer multiple of the initial tile size.
    - cls_cuda (Any, optional): Explicitly specifies a specific model base class. The default value is None, indicating that the model base class is selected automatically without explicit specification. The available base classes for specification are entrance.ApprxPICPyTorchSolution2b, entrance.ApprxPICPyTorchCuda, and entrance.ApprxPICPyTorch.
    - compile_torch (bool, optional): Boolean flag to indicate whether the torch should be compiled. Default is False.

    Returns:
        Any: The successfully loaded torch.nn.model. Returns None if the device is not supported

    """
    if device_type == 'cpu':
        return entrance.load_approach_model(device_type, '', instances_path, cls_cuda, k_dimension, n_dimension, **('device_id', 'gpu_arch_type', 'instances_path', 'cls_cuda', 'k_dimension', 'n_dimension'))
# WARNING: Decompyle incomplete


def pace2_8x8and16x16mode_matmul(x = None, y = None, pace2_instance = None, seed = (None, 'uint4'), inputType = {
    'x': torch.tensor,
    'y': torch.tensor,
    'pace2_instance': object }):
    '''
    Perform matrix multiplication using the pace2 8X8 omac instance in both 8x8 and 16x16 modes.

    This function calls the pace2 instance four times with the same input tensors x and y,
    and averages the results to produce the final output. The results are scaled by a factor of 2/8.

    Parameters:
    - x (torch.Tensor): The first input tensor.
    - y (torch.Tensor): The second input tensor.
    - pace2_instance (object): The 8X8 instance of the pace2 class for matrix multiplication.
    - seed (int, optional): Random seed for reproducibility. Default is None.
    - inputType (str, optional): The type of input data. Default is "uint4".

    Returns:
    - torch.Tensor: The resulting tensor after performinUnsupported opcode: LOAD_ASSERTION_ERROR (101)
Unsupported opcode: LOAD_ASSERTION_ERROR (101)
Unsupported opcode: LOAD_ASSERTION_ERROR (101)
Unsupported opcode: LOAD_ASSERTION_ERROR (101)
Unsupported opcode: LOAD_ASSERTION_ERROR (101)
g the matrix multiplication.
    '''
    pass
# WARNING: Decompyle incomplete


def pace2_32x32mode_matmul(x = None, y = None, pace2_instance = None, seed = (None, 'uint4'), inputType = {
    'x': torch.tensor,
    'y': torch.tensor,
    'pace2_instance': object }):
    '''
    Perform matrix multiplication using the pace2 32x32 omac instance in 32x32 mode.

    This function calls the pace2 instance twice with the same input tensors x and y,
    and averages the results to produce the final output. The results are scaled by a factor of 16/32.

    Parameters:
    - x (torch.Tensor): The first input tensor.
    - y (torch.Tensor): The second input tensor.
    - pace2_instance (object): The 32X32 instance of the pace2 class for matrix multiplication.
    - seed (int, optional): Random seed for reproducibility. Default is None.
    - inputType (str, optional): The type of input data. Default is "uint4".

    Returns:
    - torch.Tensor: The resulting tensor after performing the matrix multiplication.
    '''
    pass
# WARNING: Decompyle incomplete


def pace2_64x64mode_matmul(x = None, y = None, pace2_instance = None, seed = (None, 'uint4'), inputType = {
    'x': torch.tensor,
    'y': torch.tensor,
    'pace2_instance': object }):
    pass
# WARNING: Decompyle incomplete


def pace2_128x128mode_ising(x, y = None, pace2_8x8_instance = None, pace2_32x32_instance = None, seed = (None, 'uint4'), inputType = {
    'x': torch.tensor,
    'y': torch.tensor,
    'pace2_8x8_instance': object,
    'pace2_32x32_instance': object }):
    '''
    Perform matrix multiplication in 128x128 mode for the Ising model calculation.

    This function takes two input tensors, x and y, and performs matrix multiplication
    using both 32x32 and 8x8 instances of the pace2 class. The input tensor x is expected
    to have a shape of (1, 128), and the weight tensor y must have a shape of (128, 128).
    The function asserts the shapes of the input tensors and computes the output tensor.

    Parameters:
    - x (torch.Tensor): The input tensor with shape (1, 128).
    - y (torch.Tensor): The weight tensor with shape (128, 128).
    - pace2_8x8_instance (object): The 8X8 instance of the pace2 class.
    - pace2_32x32_instance (object): The 32X32 instance of the pace2 class.
    - seed (int, optional): Random seed for reproducibility. Default is None.
    - inputType (str, optional): The type of input data. Default is "uint4".

    Returns:
    - torch.Tensor: The resulting tensor after performing the Ising model computation.
    '''
    pass
# WARNING: Decompyle incomplete


def omac_matmul_quant(instance, x, y, scale_x = None, scale_y = None, scale_fgqs = None, inputType = ('uint4', ''), activation = {
    'instance': object,
    'x': torch.tensor,
    'y': torch.tensor,
    'scale_x': torch.tensor,
    'scale_y': torch.tensor,
    'scale_fgqs': list }):
    pass


def omac_matmul_dequant(instance, x, y, scale, inputType, laser_scale = None, gain_scale = None, noise_scale = None, freq_scale = ('uint4', 1, 1, 1, 1, False), adaptive = {
    'instance': object,
    'x': torch.tensor,
    'y': torch.tensor,
    'scale': torch.tensor,
    'laser_scale': float,
    'gain_scale': float,
    'noise_scale': float,
    'freq_scale': float,
    'adaptive': bool,
    'return': torch.tensor }):
    pass
# WARNING: Decompyle incomplete


def pace3_32x128mode_fp16int4_matmul(instance, x, y, xscale, xbias, qmax, qmin = None, qescale = None, quanttype = None, dequant = ('static', False, False), adaptive = {
    'instance': object,
    'x': torch.tensor,
    'y': torch.tensor,
    'xscale': torch.tensor,
    'xbias': torch.tensor,
    'qmax': torch.tensor,
    'qmin': torch.tensor,
    'qescale': torch.tensor }):
    '''
    This function performs matrix multiplication with quantization and scaling.
    '''
    (b, m, k, n) = (x.shape[0], x.shape[1], x.shape[2], y.shape[3])
# WARNING: Decompyle incomplete


class Pace2Param(ctypes.Structure):
    _fields_ = [
        ('B', ctypes.c_int),
        ('M', ctypes.c_int),
        ('K', ctypes.c_int),
        ('N', ctypes.c_int),
        ('TensorAddr1', ctypes.c_int),
        ('TensorAddr2', ctypes.c_int),
        ('TensorAddr3', ctypes.c_int),
        ('IsOmac', ctypes.c_int),
        ('OmacTilemode', ctypes.c_int),
        ('WeightLoadmode', ctypes.c_int),
        ('RequantEnable', ctypes.c_int),
        ('ReluEnable', ctypes.c_int),
        ('LerpEnable', ctypes.c_int),
        ('RoundMode', ctypes.c_int),
        ('PerTensorMult', ctypes.c_int),
        ('PerTensorBias', ctypes.c_int),
        ('PerTensorShift', ctypes.c_int),
        ('Tensor1DataLayout', ctypes.c_int),
        ('Tensor1DataType', ctypes.c_int),
        ('Tensor2DataLayout', ctypes.c_int),
        ('Tensor2DataType', ctypes.c_int),
        ('Tensor3DataLayout', ctypes.c_int),
        ('Tensor3DataType', ctypes.c_int)]


def pace2_matmul_with_hardware_impl(activation = None, weight = None, parameters = None):
    '''
    使用硬件加速的pace2 matmul
    Args:
        activation (torch.tensor): 激活张量
        weight (torch.tensor): 权重张量
        parameters (Pace2Param): pace2参数
    Returns:
        torch.tensor: 输出张量
    '''
    pass

