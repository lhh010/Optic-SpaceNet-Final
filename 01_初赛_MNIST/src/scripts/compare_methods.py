"""Visualization script comparing STE, LSQ+, and DSQ quantization curves."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt


def ste_quantize(x, scale, num_bits=4, is_signed=True):
    qmin = -(2 ** (num_bits - 1)) if is_signed else 0
    qmax = (2 ** (num_bits - 1) - 1) if is_signed else 2**num_bits - 1
    x_q = np.clip(np.round(x / scale), qmin, qmax)
    return x_q * scale


def lsq_quantize(x, scale, zero_point=0, num_bits=4, is_signed=True):
    qmin = -(2 ** (num_bits - 1)) if is_signed else 0
    qmax = (2 ** (num_bits - 1) - 1) if is_signed else 2**num_bits - 1
    x_int = np.clip(np.round(x / abs(scale)) + zero_point, qmin, qmax)
    return (x_int - zero_point) * abs(scale)


def dsq_quantize(x, scale, temperature=5.0, num_bits=4, is_signed=True):
    qmin = -(2 ** (num_bits - 1)) if is_signed else 0
    qmax = (2 ** (num_bits - 1) - 1) if is_signed else 2**num_bits - 1
    x_scaled = x / abs(scale)
    x_floor = np.floor(x_scaled)
    x_frac = x_scaled - x_floor
    tanh_offset = np.tanh(0.5)
    soft_frac = np.tanh(temperature * (x_frac - 0.5)) / tanh_offset * 0.5 + 0.5
    x_soft = np.clip(x_floor + soft_frac, qmin, qmax)
    return x_soft * abs(scale)


def plot_comparison():
    x = np.linspace(-2, 2, 1000)
    scale = 0.3
    y_ste = ste_quantize(x, scale)
    y_lsq = lsq_quantize(x, scale)
    y_dsq = dsq_quantize(x, scale)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.plot(x, x, "k--", label="Identity", alpha=0.5)
    ax1.plot(x, y_ste, "b-", label="STE", linewidth=2)
    ax1.plot(x, y_lsq, "r-", label="LSQ+", linewidth=2)
    ax1.plot(x, y_dsq, "g-", label="DSQ", linewidth=2)
    ax1.set_title("Quantization Functions")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(x, np.abs(y_ste - x), "b-", label="STE", linewidth=2)
    ax2.plot(x, np.abs(y_lsq - x), "r-", label="LSQ+", linewidth=2)
    ax2.plot(x, np.abs(y_dsq - x), "g-", label="DSQ", linewidth=2)
    ax2.set_title("Quantization Error")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    temps = [1, 3, 5, 10, 20]
    colors = plt.cm.viridis(np.linspace(0, 1, len(temps)))
    for temp, color in zip(temps, colors):
        ax3.plot(x, dsq_quantize(x, scale, temp), color=color, label=f"T={temp}", linewidth=2)
    ax3.plot(x, y_ste, "k--", label="Hard", alpha=0.5, linewidth=2)
    ax3.set_title("DSQ Temperature Effect")
    ax3.legend(ncol=2)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.axis("off")
    table_data = [
        ["Feature", "STE", "LSQ+", "DSQ"],
        ["Scale", "Static", "Learned", "Learned"],
        ["Gradient", "STE", "LSQ", "Soft"],
        ["Zero-point", "Fixed 0", "Learned", "Fixed 0"],
        ["Annealing", "No", "No", "Yes"],
    ]
    table = ax4.table(cellText=table_data, cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(4):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")
    for i in range(1, len(table_data)):
        color = "#D9E1F2" if i % 2 == 0 else "white"
        for j in range(4):
            table[(i, j)].set_facecolor(color)
    ax4.set_title("Method Comparison", pad=20)

    plt.tight_layout()
    plt.savefig("quantization_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: quantization_comparison.png")


if __name__ == "__main__":
    print("Generating quantization method comparison...")
    try:
        plot_comparison()
    except Exception as e:
        print(f"Plotting failed (maybe no GUI): {e}")
