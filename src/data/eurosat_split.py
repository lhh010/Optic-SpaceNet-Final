"""
================================================================================
 EuroSAT 数据划分 —— 单一数据源 (Single Source of Truth)
================================================================================
 全仓库所有训练脚本 (load_eurosat_data) 与推理脚本 (load_test_data) 必须通过
 本模块获取 train/val/test 索引, 确保:
   1. train / val / test 三者两两互斥;
   2. 三者并集 = 全集;
   3. 所有脚本拿到完全一致的划分 (同一 seed + 同一 ratio)。

 历史背景: 此前 split 逻辑在 9+ 个文件里各自复制, 导致 test 段落入 train 段
 (test < train, 100% 泄漏), 见 EXPERIMENTS.md Bug #11。本模块彻底消除该类 bug。

 划分定义 (与历史 load_test_data 的索引完全一致, 保证已存权重兼容):
   shuffled = RandomState(seed).shuffle(list(range(n)))
   val   = shuffled[:val_size]                       (模型选择)
   test  = shuffled[val_size : val_size+test_size]   (推理独立测试, 不参与训练!)
   train = shuffled[val_size+test_size :]            (梯度更新)
 默认 val_ratio=test_ratio=0.2, seed=42 → val=5400 / test=5400 / train=16200 (n=27000)。
================================================================================
"""
import numpy as np

SEED = 42
VAL_RATIO = 0.2
TEST_RATIO = 0.2


def split_indices(n, seed=SEED, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    """返回 (train_idx, val_idx, test_idx), 均为 Python list, 三者互斥且覆盖全集。

    用 list(range(n)) 做 shuffle (而非 np.arange), 以与历史代码产生逐位相同的
    排列, 从而 test 集 = 历史 load_test_data 的 indices[val_size:val_size*2], 已存
    权重无需重新对齐。
    """
    val_size = int(n * val_ratio)
    test_size = int(n * test_ratio)
    indices = list(range(n))
    np.random.RandomState(seed).shuffle(indices)

    val_idx = indices[:val_size]
    test_idx = indices[val_size:val_size + test_size]
    train_idx = indices[val_size + test_size:]

    _assert_disjoint_and_complete(val_idx, test_idx, train_idx, n)
    return train_idx, val_idx, test_idx


def _assert_disjoint_and_complete(val_idx, test_idx, train_idx, n):
    """强制不变量: 任何未来改动 (改 ratio/seed/切片) 破坏划分都会在此立即报错。"""
    s_val, s_test, s_train = set(val_idx), set(test_idx), set(train_idx)
    assert not (s_val & s_test), "BUG: val ∩ test 非空"
    assert not (s_val & s_train), "BUG: val ∩ train 非空"
    assert not (s_test & s_train), "BUG: test ∩ train 非空 (泄漏! 见 Bug #11)"
    assert len(s_val) + len(s_test) + len(s_train) == n, \
        "BUG: train/val/test 未覆盖全集 (有遗漏或重复)"


if __name__ == "__main__":
    # 自检
    n = 27000
    tr, va, te = split_indices(n)
    print(f"n={n} | train={len(tr)} val={len(va)} test={len(te)}")
    print(f"互斥+覆盖自检通过 [OK] (否则上方 assert 会抛错)")
