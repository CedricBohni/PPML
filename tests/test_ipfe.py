"""Correctness of the IPFE scheme: decrypt(encrypt(x), sk_y) == <x, y>."""

import random

import numpy as np
import pytest
import torch

from src.cryptography.cnn_ipfe import IPFE as ReferenceIPFE
from src.cryptography.optimized_cnn_ipfe import IPFE as OptimizedIPFE
from src.cryptography.optimized_cnn_ipfe import decrypt_patches_batch
from src.cryptography.prime_finder import prime_finder
from src.utils.math_helper import is_prime

P = 1721257  # prime used for 3x3 kernels in the notebooks


def random_vectors(rng, length, n):
    """Pixel patches in [0, 255] and scaled signed weights small enough for P."""
    bound = (P // 2) // (255 * length)
    xs = [[rng.randrange(256) for _ in range(length)] for _ in range(n)]
    ys = [[rng.randint(-bound, bound) for _ in range(length)] for _ in range(n)]
    return xs, ys


@pytest.fixture(autouse=True)
def _seed():
    random.seed(1337)


@pytest.mark.parametrize("scheme", [ReferenceIPFE, OptimizedIPFE])
@pytest.mark.parametrize("length", [4, 9, 25])
def test_decrypt_recovers_signed_inner_product(scheme, length):
    ipfe = scheme(P)
    ipfe.setup(length)
    rng = random.Random(length)
    for x, y in zip(*random_vectors(rng, length, 10)):
        ct = ipfe.encrypt(x)
        sk_y = ipfe.key_derive(y)
        assert ipfe.decrypt(ct, sk_y, y) == sum(a * b for a, b in zip(x, y))


def test_ciphertext_is_randomized_and_hides_plaintext():
    ipfe = ReferenceIPFE(P)
    ipfe.setup(9)
    x = [0, 133, 254, 9, 205, 248, 126, 254, 182]
    ct_a, ct_b = ipfe.encrypt(x), ipfe.encrypt(x)
    assert ct_a != ct_b  # fresh randomness per encryption
    assert ct_a[1] != x


def test_wrong_length_rejected():
    ipfe = OptimizedIPFE(P)
    ipfe.setup(4)
    with pytest.raises(ValueError):
        ipfe.encrypt([1, 2, 3])
    with pytest.raises(ValueError):
        ipfe.key_derive([1, 2, 3, 4, 5])


def test_reference_example_from_module():
    """The worked example in cnn_ipfe.py: one MNIST patch times one trained 3x3 kernel."""
    x = [0, 133, 254, 9, 205, 248, 126, 254, 182]
    w = [-0.0300, -0.0261, -0.0194, 0.0786, 0.3495, -0.0135, -0.3420, -0.1810, -0.2126]
    y = [int(v * 10000) for v in w]
    ipfe = ReferenceIPFE(P)
    ipfe.setup(9)
    result = ipfe.decrypt(ipfe.encrypt(x), ipfe.key_derive(y), y) / 10000
    assert result == pytest.approx(sum(a * b for a, b in zip(x, w)), abs=0.1)


def test_batch_decryption_matches_single_decryption():
    length = 4
    ipfe = OptimizedIPFE(P)
    ipfe.setup(length)
    rng = random.Random(7)
    xs, ys = random_vectors(rng, length, 32)
    y = ys[0]
    cts = [ipfe.encrypt(x) for x in xs]

    ct0_array = np.array([ct0 for ct0, _ in cts], dtype=np.int64)
    cts_array = np.array([c for _, c in cts], dtype=np.int64)
    batch = decrypt_patches_batch(ct0_array, cts_array, ipfe.key_derive(y), np.array(y, dtype=np.int64), ipfe.g, P)

    expected = [sum(a * b for a, b in zip(x, y)) for x in xs]
    assert batch.tolist() == expected


def test_prime_finder_bound_is_sufficient():
    torch.manual_seed(0)
    weights = torch.randn(8, 1, 3, 3) * 0.3
    p = prime_finder(weights, verbose=False)
    assert is_prime(p)

    # The worst-case patch (255 where the weight has the dominant sign) must still decrypt.
    k = weights[0, 0].flatten()
    y = [int(round(v * 10000)) for v in k.tolist()]
    pos = sum(v for v in y if v > 0)
    neg = sum(v for v in y if v < 0)
    x = [255 if (v > 0) == (pos >= -neg) and v != 0 else 0 for v in y]

    ipfe = OptimizedIPFE(p)
    ipfe.setup(9)
    assert ipfe.decrypt(ipfe.encrypt(x), ipfe.key_derive(y), y) == sum(a * b for a, b in zip(x, y))
