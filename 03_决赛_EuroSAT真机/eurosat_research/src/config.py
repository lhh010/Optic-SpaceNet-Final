"""
===============================================================================
 config.py — 实验配置 (JSON) + 哈希
===============================================================================
 每个实验 = 一个 JSON 配置文件。runner 载入后:
   - 展开默认值
   - 计算配置哈希 → run 目录名 runs/<name>_<hash8>/
   - 落盘 config.json (含 hash), 保证可复现

 容器侧零依赖 (仅 stdlib json), 本地侧同。
===============================================================================
"""
import json
import hashlib
import os
import copy


DEFAULTS = {
    # --- 数据 ---
    "data_dir": "data/EuroSAT_RGB",
    "batch_size": 64,
    "num_workers": 4,
    "aug": "standard",          # standard | strong | none
    "val_split": 0.2,
    "seed": 42,
    # --- 训练 ---
    "epochs": 80,
    "optimizer": "adamw",       # adamw | sgd | muon
    "lr": 0.001,
    "weight_decay": 5e-4,
    "warmup_epochs": 5,
    "min_lr_ratio": 0.01,
    "label_smoothing": 0.05,
    "swa": False,               # SWA: 最后 k 个 epoch 权重平均
    "swa_start_frac": 0.75,     # SWA 起点 (epochs 比例)
    "ema_decay": 0.0,           # EMA: 0 = 关闭
    "grad_clip": 0.0,           # 0 = 关闭
    "tier": "T1",               # T0=FP32 短训 (候选淘汰) | T1=全 QAT
    "t0_epochs": 20,            # T0 时的训练轮数
    # --- 模型 ---
    "arch": "minivgg_gap",      # minivgg_gap | search | custom
    "channels": [32, 48, 72, 96],
    "num_classes": 10,
    "bias": False,
    "stem_stride": 2,
    # --- QAT ---
    "qat": True,
    "qat_version": "v5",
    "weight_bits": 8,
    "act_bits": 8,
    "output_bits": 12,
    "first_conv_fp32": False,   # ★ Model 4 全光计算, 无 stem 特判 (用户决策)
    # --- 噪声 (Gazelle 逆向标定) ---
    "noise": True,
    "weight_noise": True,       # DAC ENOB 量化噪声 (v4 已有)
    "output_noise": True,       # ★ v5 修复: TIA+ADC 输出噪声注入 (v4 死代码)
    "output_noise_ratio": 0.0457,  # ★ osim random_benchmark 标定: delta_std/ideal_std
    "output_quant": True,       # ★ ADC 12-bit 输出量化
    "dac_enob": 7.5,
    "tia_noise_std": 5.34e-4,
    "adc_lsb": 0.00147,
    "noise_std_ratio": 0.0016,  # v4 权重噪声标定 (DAC ENOB 7.5)
    # --- 评测 ---
    "eval_metrics": ["acc", "macro_f1", "per_class_f1", "ece"],
    "osim_quick": 0,            # T2: osimulator q500 (仅容器内)
    # --- 输出 ---
    "run_dir": "runs",
    "name": "exp",
}


def load_config(path):
    """载入 JSON 配置并补全默认值。返回 (config dict, hash str)。"""
    with open(path, "r") as f:
        cfg = json.load(f)
    merged = copy.deepcopy(DEFAULTS)
    merged.update(cfg)
    # 强制 name 来自配置 (若未给, 用文件名)
    if "name" not in cfg or not cfg.get("name"):
        merged["name"] = os.path.splitext(os.path.basename(path))[0]
    # 计算 hash
    h = hashlib.sha256(json.dumps(merged, sort_keys=True).encode()).hexdigest()[:8]
    merged["_hash"] = h
    return merged, h


def make_run_dir(cfg, base_dir=None):
    """创建 runs/<name>_<hash>/ 目录, 返回路径。"""
    base = base_dir or cfg.get("run_dir", "runs")
    run_dir = os.path.join(base, f"{cfg['name']}_{cfg['_hash']}")
    os.makedirs(run_dir, exist_ok=True)
    # 落盘 config.json
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    return run_dir
