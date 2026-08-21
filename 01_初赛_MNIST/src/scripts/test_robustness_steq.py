"""STE baseline robustness test script."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

from src.utils.io import load_weights_and_params
from src.inference.robustness import add_noise_to_weights, run_robustness_test
from src.inference.numpy_steq import run_inference


def make_inference_fn(w1, w2, w3):
    def inference_fn(images, labels, noise_level):
        w1_noisy = add_noise_to_weights(w1, noise_level)
        w2_noisy = add_noise_to_weights(w2, noise_level)
        w3_noisy = add_noise_to_weights(w3, noise_level)
        acc, _, _, _ = run_inference(images, labels, w1_noisy, w2_noisy, w3_noisy)
        return acc

    return inference_fn


def main():
    weights_dir = os.path.join(os.path.dirname(__file__), "../../artifacts/ste")
    w1, w2, steq_params = load_weights_and_params("steq", data_dir=weights_dir)
    if not steq_params or "w3" not in steq_params:
        raise FileNotFoundError("Missing `w3_int4.npy` for 3-layer STE robustness test.")
    inference_fn = make_inference_fn(w1, w2, steq_params["w3"])
    results = run_robustness_test("STE", inference_fn)
    out_path = os.path.join(os.path.dirname(__file__), "../../artifacts/robustness/robustness_test_steq.npy")
    np.save(out_path, results)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
