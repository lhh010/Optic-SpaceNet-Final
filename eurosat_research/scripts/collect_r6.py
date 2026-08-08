#!/usr/bin/env python3
"""collect_r6.py — 汇总 runs/ 下 r6_* 实验结果为一览表 (容器内运行)。"""
import json, glob, sys, os

RUNS = "/workspace/Ltsimulator-test/auto_research/runs"
CLASSES = ['AnnualCrop','Forest','HerbVeg','Highway','Industrial',
           'Pasture','PermCrop','Residential','River','SeaLake']

rows = []
for d in sorted(glob.glob(os.path.join(RUNS, "r6_*"))):
    sj = os.path.join(d, "summary.json")
    if not os.path.exists(sj):
        rows.append((os.path.basename(d), None))
        continue
    s = json.load(open(sj))
    rows.append((s, d))

print(f"{'run':28s} {'MACs':>6s} {'params':>7s} {'val':>7s} {'TEST':>7s} {'F1':>7s}")
print("-" * 70)
table = []
for s, d in rows:
    if s is None:
        print(f"{os.path.basename(d):28s} (running / no summary)")
        continue
    name = f"{s['name']}_{s['hash']}"
    val = s['best_val_acc']*100
    test = s['test']['accuracy']*100
    f1 = s['test']['macro_f1']*100
    print(f"{name:28s} {s['macs']/1e6:5.2f}M {s['params']/1e3:6.1f}K "
          f"{val:6.2f}% {test:6.2f}% {f1:6.2f}%")
    table.append(s)

# per-class: 对照 ctrl, 看 Highway/River
ctrl = next((s for s in table if s['name'] == 'r6_ctrl'), None)
if ctrl and '--perclass' in sys.argv:
    print("\nper-class F1 (test), Δ vs r6_ctrl:")
    cf = ctrl['test']['per_class_f1']
    hdr = f"{'run':22s} " + " ".join(f"{c[:6]:>7s}" for c in CLASSES)
    print(hdr)
    for s in table:
        if s['name'] == 'r6_ctrl':
            continue
        f1 = s['test']['per_class_f1']
        deltas = " ".join(f"{(a-b)*100:+7.2f}" for a, b in zip(f1, cf))
        print(f"{s['name']:22s} {deltas}")
