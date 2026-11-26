"""Export the default CNN and sample MNIST digits for the in-browser demo in ``site/``.

Writes ``site/model.json`` with
- the model weights (batch norm folded into a per-channel scale and shift),
- the IPFE parameters (prime, weight scale) from ``configs/base.yaml``,
- 30 MNIST test digits as uint8 pixel arrays,
- PyTorch reference logits for those digits, used by ``site/tests/check.mjs``.

Usage::

    uv run python scripts/export_web_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
N_SAMPLES = 30
DECIMALS = 7


def rounded(t: torch.Tensor) -> list:
    return torch.round(t.double() * 10**DECIMALS).div(10**DECIMALS).tolist()


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from src.cnn.models import PlainCNN
    from src.cnn.utils import find_checkpoint
    from src.cryptography.prime_finder import WEIGHT_SCALE, prime_finder

    with open(ROOT / "configs" / "base.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["save"]["models_dir"] = str(ROOT / cfg["save"]["models_dir"])
    ckpt = find_checkpoint(cfg)
    model = PlainCNN(cfg).eval()
    model.load_from_checkpoint(str(ckpt))
    bb = model.backbone

    prime = cfg["ipfe"]["prime"]
    assert prime >= prime_finder(bb.conv1.weight, verbose=False), "configured prime is too small"

    layers = []
    m = cfg["model"]
    for i in range(len(m["c"])):
        conv, bn = getattr(bb, f"conv{i + 1}"), getattr(bb, f"bn{i + 1}")
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        shift = bn.bias - bn.running_mean * scale
        layers.append(
            {
                "weight": rounded(conv.weight.detach()),
                "bias": rounded(conv.bias.detach()),
                "bnScale": rounded(scale.detach()),
                "bnShift": rounded(shift.detach()),
                "stride": conv.stride[0],
                "padding": conv.padding[0],
                "pool": hasattr(bb, f"pool{i + 1}"),
            }
        )

    test_set = datasets.MNIST(ROOT / "data", train=False, download=True, transform=transforms.PILToTensor())
    images = torch.stack([test_set[i][0] for i in range(N_SAMPLES)])
    labels = [int(test_set[i][1]) for i in range(N_SAMPLES)]
    with torch.no_grad():
        logits = model(images)

    out = {
        "source": ckpt.name,
        "ipfe": {"prime": prime, "weightScale": WEIGHT_SCALE},
        "conv": layers,
        "fc": [
            {"weight": rounded(bb.fc1.weight.detach()), "bias": rounded(bb.fc1.bias.detach())},
            {"weight": rounded(bb.fc2.weight.detach()), "bias": rounded(bb.fc2.bias.detach())},
        ],
        "samples": [{"label": lab, "pixels": img.flatten().tolist()} for lab, img in zip(labels, images)],
        "referenceLogits": rounded(logits),
    }
    dest = ROOT / "site" / "model.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    acc = sum(int(p == y) for p, y in zip(logits.argmax(1).tolist(), labels))
    print(f"Wrote {dest} ({dest.stat().st_size / 1024:.0f} KiB) from {ckpt.name}; reference accuracy {acc}/{N_SAMPLES}")


if __name__ == "__main__":
    main()
