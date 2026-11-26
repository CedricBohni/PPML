"""Time encrypted inference for every decryption strategy on the same MNIST test images.

Unlike the full benchmark suite (``src/main.py``), this script keeps everything fixed except the
strategy: same images, same keys (seeded), JIT warmed up before timing, encryption excluded.
It also checks that every strategy predicts the same digits.

Usage::

    uv run python scripts/benchmark_strategies.py              # 16 images, both models
    uv run python scripts/benchmark_strategies.py --images 64
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cnn.models import IPFECNN  # noqa: E402
from src.cnn.utils import find_checkpoint  # noqa: E402
from src.cryptography.optimized_cnn_ipfe import decrypt_patches_batch  # noqa: E402
from src.cryptography.prime_finder import prime_finder  # noqa: E402

STRATEGIES = {
    "sequential": None,
    "Kernel threads": "kernel_parallelization",
    "Kernel + patch threads": "kernel_patches_parallelization",
    "Batched (Numba)": "batch_parallelization",
    "Batched (Numba) + kernel threads": "batch_kernels_parallelization",
}
#: First-layer stride of the committed ``k3_conv8_16_32_64_stride{s}_pad1`` checkpoints.
MODELS = {"3×3, stride 3 (100 patches)": 3, "3×3, stride 1 (784 patches)": 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=int, default=16)
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.stdout.reconfigure(encoding="utf-8")  # "×" in the table on Windows consoles

    with open("configs/base.yaml") as f:
        base = yaml.safe_load(f)
    base["device"] = "cpu"
    test_set = datasets.MNIST("./data", train=False, download=True, transform=transforms.PILToTensor())
    images = torch.stack([test_set[i][0] for i in range(args.images)])

    # Compile the Numba kernels before any timing.
    decrypt_patches_batch(
        np.array([1], dtype=np.int64), np.array([[1, 1]], dtype=np.int64), 0, np.array([0, 0], dtype=np.int64), 2, 5
    )

    print(f"{args.images} MNIST test images, {os.cpu_count()} logical CPUs\n")
    print("| Model | Strategy | Seconds | ms / image | Speed-up |")
    print("|---|---|--:|--:|--:|")
    for model_name, stride in MODELS.items():
        cfg0 = copy.deepcopy(base)
        cfg0["model"]["stride"][0] = stride
        ckpt = find_checkpoint(cfg0)
        conv1 = torch.load(ckpt, map_location="cpu")["backbone.conv1.weight"]
        cfg0["ipfe"]["prime"] = prime_finder(conv1, verbose=False)

        baseline, reference = None, None
        for label, strategy in STRATEGIES.items():
            cfg = copy.deepcopy(cfg0)
            cfg["model"]["name"] = "ipfe"
            for s in STRATEGIES.values():
                if s:
                    cfg["optimizations"][s] = s == strategy
            random.seed(0)
            model = IPFECNN(cfg).eval()
            model.load_from_checkpoint(str(ckpt))
            encrypted = model.encrypt_data(images)

            with torch.no_grad():
                model(encrypted[:1])  # warm-up
                t0 = time.perf_counter()
                pred = model(encrypted).argmax(1)
                seconds = time.perf_counter() - t0

            baseline = baseline or seconds
            if reference is None:
                reference = pred
            assert torch.equal(pred, reference), f"{label} predicts differently from the sequential baseline"
            print(
                f"| {model_name} | {label} | {seconds:.2f} | {1000 * seconds / args.images:.0f} "
                f"| {baseline / seconds:.1f}× |",
                flush=True,
            )


if __name__ == "__main__":
    main()
