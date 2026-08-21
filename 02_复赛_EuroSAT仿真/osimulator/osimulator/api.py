import os
import torch
import torch.nn.functional as F
import entrance

osim_path = os.path.dirname(os.path.abspath(__file__))

gpu_device_type_dicts = {"sm80": ["A100", "A800"],
                         "sm86": ["3080", "3090", "A2000", "A3000", "A4000", "A5000", "A6000", "A40", "3060", "3070", "3050", "A10", "A16", "A40"],
                         "sm89": ["4090", "4080", "6000", "L20", "L40", "L4"],
                         "sm90": ["H100", "H200", "H20"],
                         "gpu": [""]}

def load_gazelle_model():
    """Loads the Gazelle model.
    Returns:
        GazelleModel: An instance of the Gazelle model.
    """
    return entrance.load_approach_model(device_id="cpu", gpu_arch_type="", 
                                        instances_path=osim_path + "/models/gazelle/8X2_8a8w12o_dacenob7.5_power0.015_noise9e-11_10mclock_mean-1.49_std5.31", cls_cuda=None)
