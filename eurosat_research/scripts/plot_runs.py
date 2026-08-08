"""
===============================================================================
 plot_runs.py — 本地实验对比可视化
===============================================================================
 用法:
   python plot_runs.py runs/                 # 汇总所有 run 的 summary
   python plot_runs.py runs/ --curves val_acc  # 画对比曲线
   python plot_runs.py runs/ --to-tb out/    # JSONL → TensorBoard event
===============================================================================
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_runs(runs_dir):
    runs = []
    for summary_path in sorted(glob.glob(os.path.join(runs_dir, "*/summary.json"))):
        with open(summary_path) as f:
            summary = json.load(f)
        metrics_path = os.path.join(os.path.dirname(summary_path), "metrics.jsonl")
        curves = []
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                for line in f:
                    curves.append(json.loads(line))
        runs.append({"summary": summary, "curves": curves,
                     "dir": os.path.dirname(summary_path),
                     "name": summary.get("name", "?")})
    return runs


def print_table(runs):
    print(f"{'name':<28s} {'params':>8s} {'MACs(M)':>8s} {'best_val':>8s} "
          f"{'test_acc':>8s} {'test_f1':>8s} {'ece':>7s} {'ep':>4s}")
    print("-" * 90)
    for r in runs:
        s = r["summary"]
        test = s.get("test", {})
        print(f"{s['name']:<28s} {s['params']:>8,} {s['macs']/1e6:>8.2f} "
              f"{s['best_val_acc']:>8.2%} {test.get('accuracy', 0):>8.2%} "
              f"{test.get('macro_f1', 0):>8.2%} {test.get('ece', 0):>7.4f} "
              f"{s.get('best_epoch', 0):>4d}")


def plot_curves(runs, metric, out_png):
    plt.figure(figsize=(10, 6))
    for r in runs:
        if not r["curves"]:
            continue
        ep = [c["epoch"] for c in r["curves"]]
        val = [c.get(metric, np.nan) for c in r["curves"]]
        plt.plot(ep, val, label=r["name"], linewidth=1.5)
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(f"{metric} comparison")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"saved {out_png}")


def to_tensorboard(runs, out_dir):
    """JSONL → tfevents (本地 tensorboard 已装 2.20)。"""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("torch/tensorboard not available locally, skipping")
        return
    for r in runs:
        if not r["curves"]:
            continue
        name = r["name"]
        writer = SummaryWriter(os.path.join(out_dir, name))
        for c in r["curves"]:
            ep = c["epoch"]
            for k, v in c.items():
                if k != "epoch":
                    writer.add_scalar(k, v, ep)
        s = r["summary"]
        for phase in ("val", "test"):
            if phase in s:
                for k, v in s[phase].items():
                    if k in ("accuracy", "macro_f1", "ece", "loss"):
                        writer.add_scalar(f"final/{phase}_{k}", v, 0)
        writer.close()
    print(f"tensorboard events written to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--curves", default=None, help="metric name e.g. val_acc")
    ap.add_argument("--to-tb", default=None)
    ap.add_argument("--out", default="docs/figures/runs_compare.png")
    args = ap.parse_args()

    runs = load_runs(args.runs_dir)
    print(f"found {len(runs)} runs")
    print_table(runs)

    if args.curves:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        plot_curves(runs, args.curves, args.out)
    if args.to_tb:
        to_tensorboard(runs, args.to_tb)


if __name__ == "__main__":
    main()
