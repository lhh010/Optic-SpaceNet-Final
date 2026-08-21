Unsupported opcode: LOAD_ASSERTION_ERROR (101)
# Source Generated with Decompyle++
# File: PostProcess.cpython-39.pyc (Python 3.9)

import torch
import triton
from triton.language import language as tl

def dequant_kernel(a_ptr, s_ptr, r_ptr, B, B_POW, M, N, stride_ab, stride_am, stride_an, stride_sm, stride_sn, stride_rm = None, stride_rn = triton.autotune([
    triton.Config({
        'BLOCK_SIZE_M': 16,
        'BLOCK_SIZE_N': 16 }, 3, 8, **('num_stages', 'num_warps')),
    triton.Config({
        'BLOCK_SIZE_M': 32,
        'BLOCK_SIZE_N': 32 }, 3, 8, **('num_stages', 'num_warps')),
    triton.Config({
        'BLOCK_SIZE_M': 64,
        'BLOCK_SIZE_N': 64 }, 3, 8, **('num_stages', 'num_warps'))], [
    'M',
    'N',
    'B'], **('configs', 'key')), BLOCK_SIZE_M = triton.jit, BLOCK_SIZE_N = {
    'B': tl.constexpr,
    'B_POW': tl.constexpr,
    'M': tl.constexpr,
    'N': tl.constexpr,
    'stride_ab': tl.constexpr,
    'stride_am': tl.constexpr,
    'stride_an': tl.constexpr,
    'stride_sm': tl.constexpr,
    'stride_sn': tl.constexpr,
    'stride_rm': tl.constexpr,
    'stride_rn': tl.constexpr,
    'BLOCK_SIZE_M': tl.constexpr,
    'BLOCK_SIZE_N': tl.constexpr }):
    pid = tl.program_id(0, **('axis',))
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offset_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset_b = tl.arange(0, B_POW)
    a_ptrs = a_ptr + offset_b[(:, None, None)] * stride_ab + offset_m[(None, :, None)] * stride_am + offset_n[(None, None, :)] * stride_an
    s_ptrs = s_ptr + offset_m[(:, None)] * stride_am + offset_n[(None, :)] * stride_an
    r_ptrs = r_ptr + offset_m[(:, None)] * stride_am + offset_n[(None, :)] * stride_an
    mask_b = offset_b < B
    mask_m = offset_m < M
    mask_n = offset_n < N
    maska = mask_b[(:, None, None)] & mask_m[(None, :, None)] & mask_n[(None, None, :)]
    masks = mask_m[(:, None)] & mask_n[(None, :)]
    data_a = tl.load(a_ptrs, maska, 0, **('mask', 'other'))
    data_s = tl.load(s_ptrs, masks, 0, **('mask', 'other')).reshape((1, BLOCK_SIZE_M, BLOCK_SIZE_N))
    rs = tl.sum(data_a * data_s, 0).to(tl.float16)
    tl.store(r_ptrs, rs, masks, **('mask',))

dequant_kernel = None(None(dequant_kernel))

def trition_dequant(Output, Scale):
    pass
# WARNING: Decompyle incomplete

