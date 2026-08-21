Unsupported opcode: DICT_MERGE (213)
Unsupported opcode: DICT_MERGE (213)
# Source Generated with Decompyle++
# File: GPUMemoryTracker.cpython-39.pyc (Python 3.9)

import torch
import torch.jit as torch
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

def analyze_memory_detailed(model = None, device = None, top_n = None, **forward_kwargs):
    """
    Detailed analysis of model memory usage (including static and dynamic)
    Args:
        model: The model instance be instantiated by load_approach_model()
        device: Device indicate which GPU device will be profiled
        top_n: return top N operations with highest memory usage
        forward_kwargs: input args of the model
    
    Returns:
        dict: {
            'model_parameters': {...},
            'model_buffers': {...},
            'peak_memory': int,
            'operations': [{...}, ...]
        }
    """
    print('\n================================================================================')
    print('Static Memory Analysis')
    print('================================================================================')
    static_memory = { }
    total_params = 0
# WARNING: Decompyle incomplete


def analyze_operations_memory_topn(model = None, device = None, top_n = None, **forward_kwargs):
    '''
    Analyze memory usage for each operation and sort by Top-N
    
    Args:
        model: The model instance be instantiated by load_approach_model()
        device: Device indicate which GPU device will be profiled
        top_n: return top N operations with highest memory usage
        forward_kwargs: input args of the model
    
    Returns:
        list: [(operation_name, memory_mb, count, avg_time_us), ...]
    '''
    pass
# WARNING: Decompyle incomplete

