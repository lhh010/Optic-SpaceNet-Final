"""Shared robustness test framework.

References:
    - src_raw/test_robustness.py
    - src_dsqlsq/test_robustness_lsq_plus.py
    - src_dsqlsq/test_robustness_dsq.py
"""

import os
import time
import numpy as np


def add_noise_to_weights(w, noise_level=0.1):
    """Simulate device drift by adding Gaussian noise to INT4 weights."""
    noise = np.random.normal(0, noise_level, w.shape)
    w_noisy = w + np.round(noise)
    w_noisy = np.clip(w_noisy, -8, 7).astype(np.int32)
    return w_noisy


def load_test_data():
    """Load MNIST test images and labels from the project data directory."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_dir = os.path.join(project_root, "data", "processed")
    images_path = os.path.join(data_dir, "test_images.npy")
    labels_path = os.path.join(data_dir, "test_labels.npy")

    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        raise RuntimeError(f"Cannot load test data from `{data_dir}`")

    images = np.load(images_path)
    labels = np.load(labels_path)
    return images, labels


def run_robustness_test(method_name, inference_fn, noise_levels=None, num_runs=3):
    """Run robustness test across a range of noise levels.

    Args:
        method_name: str for display (e.g., 'LSQ+').
        inference_fn: Callable(images, labels, w1_noisy, w2_noisy) -> accuracy.
        noise_levels: List of noise levels to test.
        num_runs: Number of repetitions per noise level for averaging.

    Returns:
        results: list of dicts with keys 'noise_level', 'accuracy', 'std', 'time'.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    print("=" * 60)
    print(f"{method_name} Robustness Test")
    print("=" * 60)

    images, labels = load_test_data()
    print(f"Test samples: {len(labels)}")

    print("\n" + "=" * 60)
    print("Starting robustness test...")
    print("=" * 60)
    print(f"{'Noise Level':<12} | {'Accuracy':<12} | Status")
    print("-" * 60)

    results = []
    for noise in noise_levels:
        start = time.time()
        accuracies = []
        for _ in range(num_runs):
            acc = inference_fn(images, labels, noise)
            accuracies.append(acc)

        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        elapsed = time.time() - start
        status = "OK" if avg_acc >= 85.0 else "FAIL"
        print(f"{noise:<12.2f} | {avg_acc:>6.2f}%±{std_acc:>4.2f} | {status}")

        results.append(
            {"noise_level": noise, "accuracy": avg_acc, "std": std_acc, "time": elapsed}
        )

    tolerance = 0.0
    for r in results:
        if r["accuracy"] >= 85.0:
            tolerance = r["noise_level"]

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Noise tolerance (>=85% accuracy): {tolerance:.2f}")
    print(f"Accuracy at noise=0.5: {next(r for r in results if r['noise_level'] == 0.5)['accuracy']:.2f}%")
    print("=" * 60)
    return results
