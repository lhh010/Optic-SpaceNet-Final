"""
===============================================================================
 metrics.py — 充分评估 (不止 acc/loss)
===============================================================================
   - acc:        准确率
   - macro_f1:   F1 macro (类别均衡)
   - per_class_f1: 每类 F1 (定位类别瓶颈)
   - ece:        Expected Calibration Error (置信度校准)
   - confusion:  混淆矩阵 (JSON 序列化友好)
===============================================================================
"""
import numpy as np
import torch


def evaluate_full(model, loader, device, num_classes=10):
    """完整评测: 返回 acc / macro_f1 / per_class_f1 / ece / confusion / loss。"""
    model.eval()
    model.to(device)
    all_preds, all_labels, all_conf, total_loss = [], [], [], 0.0
    criterion = torch.nn.CrossEntropyLoss()
    n_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            conf, preds = probs.max(dim=1)
            total_loss += loss.item() * images.size(0)
            n_samples += images.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_conf.append(conf.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    conf = np.concatenate(all_conf)
    return _compute_metrics(preds, labels, conf, total_loss / n_samples, num_classes)


def _compute_metrics(preds, labels, conf, avg_loss, num_classes):
    """从 numpy 数组计算全部指标。"""
    acc = float((preds == labels).mean())

    # --- per-class F1 ---
    per_class_f1 = np.zeros(num_classes)
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_f1[c] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    macro_f1 = float(per_class_f1.mean())

    # --- ECE (10 bins) ---
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.sum() > 0:
            avg_conf = conf[in_bin].mean()
            avg_acc = (preds[in_bin] == labels[in_bin]).mean()
            ece += (in_bin.sum() / len(conf)) * abs(avg_conf - avg_acc)

    # --- confusion matrix ---
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        confusion[t, p] += 1

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1.tolist(),
        "ece": float(ece),
        "loss": avg_loss,
        "confusion": confusion.tolist(),
        "n": int(len(labels)),
    }
