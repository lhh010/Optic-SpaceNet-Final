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
