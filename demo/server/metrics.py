"""Static exhibition-board metrics for the demo (see docs/SUMMARY.md)."""

METRICS = {
    "optic_ratio": 0.9065,      # 光计算 MOPs 占比
    "mops_total": 1.0511,       # 全模型 MOPs/张
    "mops_vs_model1": "150×",   # 相对 Model 1 的 MOPs 压缩
    "osim_full_acc": 0.9028,    # osim 全量 test 精度
    "osim_full_n": 5400,        # osim 全量评估样本数
    "hw_align": 0.996,          # 硬件对齐率
    "val_int8": 0.9183,         # int8 val 精度
    "params": 267944,           # 参数量
    "per_image_s": 2.5,         # 真机单张推理耗时 (秒)
}


def get_metrics():
    return dict(METRICS)


# ---- M9 / M10 静态展板指标 (真机全量口径, 见 02_验证报告 / board_validation) ----
# 光计算占比为近似值 (stem 电 + 7 光层 + head 光), 未做逐层官方 MOPs 拆分。
METRICS_M9 = {
    "optic_ratio": 0.90,
    "mops_total": 1.52,
    "mops_vs_model1": "103×",
    "osim_full_acc": 0.9443,
    "osim_full_n": 5400,
    "hw_align": 0.99,
    "val_int8": 0.9587,
    "params": 54900,
    "per_image_s": 2.0,
    "note": "M9 w075ds3 · 真机全量 5400 = 94.43% (gap −1.44)",
}

METRICS_M10 = {
    "optic_ratio": 0.90,
    "mops_total": 2.56,
    "mops_vs_model1": "61×",
    "osim_full_acc": 0.9533,
    "osim_full_n": 5400,
    "hw_align": 0.99,
    "val_int8": 0.9676,
    "params": 96600,
    "per_image_s": 3.2,
    "note": "M10 ds3pool3 · 真机全量 5400 = 95.33% (gap −1.43)",
}
