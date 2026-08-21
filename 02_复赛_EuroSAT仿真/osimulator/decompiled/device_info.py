Unsupported Node type: 12
# Source Generated with Decompyle++
# File: device_info.cpython-39.pyc (Python 3.9)

from collections import OrderedDict
import random
import os
import sys
test_analog_root = os.path.dirname(os.path.abspath(__file__)) + '/'
test_photonic_root = test_analog_root
sys.path.append(f'''{test_analog_root}/library''')
sys.path.append(f'''{test_analog_root}..''')
import entrance
import threading

def singleton(cls):
    _instance = { }
    
    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]

    return inner

DeviceInfoCollector = singleton(<NODE:12>)

def compass_forward_hook(module, b, m, k, n):
    compass_latency = entrance.gazelle_latency(b, m, k, n)
    return compass_latency

compass_forward_hook = DeviceInfoCollector()(compass_forward_hook)
