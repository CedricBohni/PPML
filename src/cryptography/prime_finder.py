"""Choose a prime modulus large enough for IPFE decryption of a given first conv layer."""

from __future__ import annotations

import numpy as np
import torch

from src.utils.math_helper import next_prime

#: Pixel values are uint8, so each patch entry lies in [0, PIXEL_MAX].
PIXEL_MAX = 255
#: Weights are scaled by this factor and rounded to integers before key derivation.
WEIGHT_SCALE = 10000


def prime_finder(weights: torch.Tensor, verbose: bool = True) -> int:
    """Return the smallest prime that safely decrypts every inner product of ``weights``.

    For each kernel, the largest possible ``|<patch, kernel>|`` is reached by setting
    pixels to 255 where the weight has the dominant sign and to 0 elsewhere. Decryption
    recovers values in ``Z_{p-1}`` and maps the upper half to negatives, so ``p`` must
    exceed twice the largest scaled inner product.

    Args:
        weights: conv weights of shape ``(out_ch, in_ch, k, k)``.
        verbose: print the intermediate bounds (as the test suite does).
    """
    x = weights.detach().cpu().numpy()

    max_pos = PIXEL_MAX * np.maximum(x, 0).sum(axis=(2, 3))
    max_neg = PIXEL_MAX * np.minimum(x, 0).sum(axis=(2, 3))
    global_max_abs = np.maximum(np.abs(max_pos), np.abs(max_neg)).max()

    scaled_max = global_max_abs * WEIGHT_SCALE
    prime_minimum = scaled_max * 2
    prime = next_prime(prime_minimum)

    if verbose:
        print("Maximum absolute inner product over all slices:", global_max_abs)
        print(f"Scaled maximum absolute inner product (x{WEIGHT_SCALE}):", scaled_max)
        print("Suggested prime minimum (scaled max * 2):", prime_minimum)
        print("Suggested prime (next prime above minimum):", prime)
    return prime
