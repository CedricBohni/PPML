"""The encrypted CNN must reproduce the plaintext CNN's predictions.

Uses ``configs/base.yaml`` (which points at a committed pretrained checkpoint) and
synthetic uint8 images, so no training or dataset download is needed.
"""

import copy
from pathlib import Path

import pytest
import torch
import yaml

from src.cnn.models import IPFECNN, PlainCNN
from src.cnn.utils import _build_model_tag_from_cfg, find_checkpoint
from src.cryptography.prime_finder import prime_finder

ROOT = Path(__file__).resolve().parent.parent
OPTIMIZATIONS = [
    "kernel_parallelization",
    "kernel_patches_parallelization",
    "batch_parallelization",
    "batch_kernels_parallelization",
]


@pytest.fixture(scope="module")
def cfg():
    with open(ROOT / "configs" / "base.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["save"]["models_dir"] = str(ROOT / cfg["save"]["models_dir"])
    cfg["device"] = "cpu"
    return cfg


@pytest.fixture(scope="module")
def images():
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (3, 1, 28, 28), generator=g, dtype=torch.uint8)


def test_model_tag_matches_checkpoint_name(cfg):
    assert _build_model_tag_from_cfg(cfg) == "k3_conv8_16_32_64_stride3_pad1_dropout0.5"
    assert find_checkpoint(cfg).exists()


def test_configured_prime_is_large_enough(cfg):
    conv1 = torch.load(find_checkpoint(cfg), map_location="cpu")["backbone.conv1.weight"]
    assert cfg["ipfe"]["prime"] >= prime_finder(conv1, verbose=False)
    assert cfg["ipfe"]["encryption_length"] == cfg["model"]["k"][0] ** 2


def test_plain_cnn_output_shape(cfg, images):
    model = PlainCNN(cfg).eval()
    model.load_from_checkpoint(str(find_checkpoint(cfg)))
    assert model(images).shape == (3, cfg["model"]["num_classes"])


def _ipfe_cfg(cfg, optimization):
    c = copy.deepcopy(cfg)
    c["model"]["name"] = "ipfe"
    c["optimizations"]["precrypted"] = True
    for opt in OPTIMIZATIONS:
        c["optimizations"][opt] = opt == optimization
    return c


@pytest.mark.parametrize("optimization", [None, *OPTIMIZATIONS])
def test_encrypted_inference_matches_plaintext(cfg, images, optimization):
    ckpt = str(find_checkpoint(cfg))
    plain = PlainCNN(cfg).eval()
    plain.load_from_checkpoint(ckpt)

    ipfe_cfg = _ipfe_cfg(cfg, optimization)
    encrypted = IPFECNN(ipfe_cfg).eval()
    encrypted.load_from_checkpoint(ckpt)

    with torch.no_grad():
        expected = plain(images)
        actual = encrypted(encrypted.encrypt_data(images))

    # Weights are quantized to 1e-4 for IPFE, so logits agree closely but not bit-exactly.
    assert torch.equal(actual.argmax(1), expected.argmax(1))
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.01)


def test_only_one_optimization_allowed(cfg):
    c = _ipfe_cfg(cfg, "batch_parallelization")
    c["optimizations"]["kernel_parallelization"] = True
    with pytest.raises(ValueError):
        IPFECNN(c)
