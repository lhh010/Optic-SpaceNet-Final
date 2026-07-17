"""Tests for demo/server/inference_local.py — local FP32 / fake-optical paths.

Uses the real weight file and a real EuroSAT test-set image (seed=42 test
segment); asserts the PathResult contract from demo/docs/api.md on both paths.
"""
import io

import pytest
import torch
from PIL import Image

from conftest import REPO_ROOT  # noqa: F401  (sys.path bootstrap side effect)
import contract
from demo.server import inference_local


def _first_test_image():
    """(jpeg bytes, label) of the first image in the clean test split."""
    from eurosat_split import split_indices
    from torchvision import datasets

    ds = datasets.ImageFolder(str(inference_local.DATA_DIR))
    assert ds.classes == inference_local.CLASSES, \
        "CLASSES must match ImageFolder alphabetical order"
    _, _, test_idx = split_indices(len(ds), seed=42, val_ratio=0.2, test_ratio=0.2)
    path, target = ds.samples[test_idx[0]]
    with open(path, "rb") as f:
        return f.read(), ds.classes[target]


@pytest.fixture(scope="module")
def test_image():
    return _first_test_image()


@pytest.fixture(scope="module")
def img_tensor(test_image):
    return inference_local.preprocess(test_image[0])


def test_preprocess_shape_and_range(test_image):
    t = inference_local.preprocess(test_image[0])
    assert t.shape == (1, 3, 64, 64)
    assert t.dtype == torch.float32
    # ImageNet-normalized: roughly zero-centered, not raw [0,1]
    assert t.min() < 0 < t.max()


def test_preprocess_accepts_arbitrary_size():
    img = Image.new("RGB", (200, 120), color=(10, 200, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    t = inference_local.preprocess(buf.getvalue())
    assert t.shape == (1, 3, 64, 64)


def test_fp32_path_contract(img_tensor, test_image):
    res = inference_local.infer_fp32(img_tensor)
    contract.check_path_result(res, "fp32-local")
    # real trained weights on a real test image: top-1 must be the true label
    # (this image is from the clean test split; FP32 val acc is ~92%, and the
    # first test image is correctly classified — pinned for determinism)
    assert res["pred"] == test_image[1]


def test_fake_optical_path_contract(img_tensor):
    res = inference_local.infer_fake(img_tensor)
    contract.check_path_result(res, "fake-optical")
    wheres = [l["where"] for l in res["layers"]]
    assert wheres == ["electronic"] + ["optical"] * 5


def test_two_paths_isomorphic(img_tensor):
    fp32 = inference_local.infer_fp32(img_tensor)
    fake = inference_local.infer_fake(img_tensor)
    assert fp32.keys() == fake.keys()
    assert [l["name"] for l in fp32["layers"]] == [l["name"] for l in fake["layers"]]
    for a, b in zip(fp32["layers"], fake["layers"]):
        assert a["shape"] == b["shape"]
        assert a["where"] == b["where"]


def test_lazy_singletons_reused():
    m1 = inference_local.get_fp32_model()
    m2 = inference_local.get_fp32_model()
    assert m1 is m2
    f1 = inference_local.get_fake_model()
    f2 = inference_local.get_fake_model()
    assert f1 is f2
