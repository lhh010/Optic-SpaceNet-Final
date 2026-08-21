"""DSQ robustness test script."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

from src.utils.io import load_weights_and_params
from src.inference.robustness import add_noise_to_weights, run_robustness_test
from src.inference.numpy_dsq import run_inference


def make_inference_fn(w1, w2, w3, quant_params):
    def inference_fn(images, labels, noise_level):
        w1_noisy = add_noise_to_weights(w1, noise_level)
        w2_noisy = add_noise_to_weights(w2, noise_level)
        w3_noisy = add_noise_to_weights(w3, noise_level)
        acc, _ = run_inference(images, labels, w1_noisy, w2_noisy, w3_noisy, quant_params)
        return acc

    return inference_fn


def main():
    weights_dir = os.path.join(os.path.dirname(__file__), "../../artifacts/dsq")
    w1, w2, quant_params = load_weights_and_params("dsq", data_dir=weights_dir)
    if "w3" not in quant_params:
        raise FileNotFoundError("Missing `w3_int4_dsq.npy` for 3-layer DSQ robustness test.")
    inference_fn = make_inference_fn(w1, w2, quant_params["w3"], quant_params)
    results = run_robustness_test("DSQ", inference_fn)
    out_path = os.path.join(os.path.dirname(__file__), "../../artifacts/robustness/robustness_test_dsq.npy")
    np.save(out_path, results)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
