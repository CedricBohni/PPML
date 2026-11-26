"""Numba-accelerated DDH-based Inner-Product Functional Encryption (IPFE).

Same scheme as :mod:`src.cryptography.cnn_ipfe`, but the hot decryption path
(modular exponentiation + baby-step giant-step discrete log) is JIT-compiled
with Numba, and :func:`decrypt_patches_batch` decrypts many ciphertexts for one
functional key in a single compiled loop.

Reference: Abdalla, Bourse, De Caro, Pointcheval,
"Simple Functional Encryption Schemes for Inner Products", PKC 2015.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
from numba import njit

from src.utils.math_helper import bsgs_numba, find_generator, mod_inv_numba, mod_pow_numba

Ciphertext = tuple[int, list[int]]


@njit
def decrypt_fast(ct0, cts, sk_y, y, p):
    """Return ``g^<x, y> mod p`` from a ciphertext ``(ct0, cts)`` and functional key ``sk_y``."""
    num = 1
    for i in range(len(cts)):
        num = (num * mod_pow_numba(cts[i], y[i] % (p - 1), p)) % p
    denom = mod_pow_numba(ct0, sk_y % (p - 1), p)
    return (num * mod_inv_numba(denom, p)) % p


@njit
def decrypt_patches_batch(ct0_array, cts_array, sk_y, y_vec, g, p):
    """Decrypt many ciphertexts for the same functional key ``(sk_y, y_vec)``.

    Args:
        ct0_array: ``(N,)`` array with the ``ct0`` component of each ciphertext.
        cts_array: ``(N, l)`` array with the ``ct_i`` components of each ciphertext.
        sk_y: functional secret key for ``y_vec``.
        y_vec: integer (scaled) weight vector of length ``l``.
        g, p: group generator and prime modulus.

    Returns:
        ``(N,)`` int64 array of signed inner products ``<x_n, y_vec>``.
    """
    num_patches = len(ct0_array)
    results = np.zeros(num_patches, dtype=np.int64)
    y_vec_int = [int(y) for y in y_vec]
    modulus = p - 1
    half = modulus // 2

    for i in range(num_patches):
        cts = [int(c) for c in cts_array[i]]
        val = decrypt_fast(int(ct0_array[i]), cts, int(sk_y), y_vec_int, p)

        ip = bsgs_numba(g, val, p, p - 1)
        if ip == -1:
            raise ValueError("Discrete log failed")

        # Exponents live in Z_{p-1}; map the upper half back to negative values.
        results[i] = ip - modulus if ip > half else ip

    return results


class IPFE:
    """DDH-based IPFE over the multiplicative group ``Z_p^*``.

    The holder of the master secret key can derive a key ``sk_y`` for a vector ``y``.
    Anyone holding ``sk_y`` learns ``<x, y>`` from an encryption of ``x``, and nothing else
    about ``x``. ``p`` must be prime and larger than twice the largest absolute inner
    product, so that signed results can be recovered.
    """

    def __init__(self, p: int):
        self.p = p
        self.g: int | None = None
        self.length: int | None = None
        self.mpk: list[int] | None = None
        self.msk: list[int] | None = None

    def setup(self, l: int) -> None:  # noqa: E741 - `l` is the vector length in the paper
        """Generate the master keys for vectors of length ``l``.

        ``msk = s = (s_1, ..., s_l)`` sampled uniformly and ``mpk = (h_i = g^{s_i})``.
        """
        self.length = l
        self.g = find_generator(self.p)
        s = [random.randrange(1, self.p - 1) for _ in range(self.length)]
        self.mpk = [pow(self.g, s_i, self.p) for s_i in s]
        self.msk = s

    def encrypt(self, x: Sequence[int]) -> Ciphertext:
        """Encrypt ``x``: ``ct0 = g^r`` and ``ct_i = h_i^r * g^{x_i}`` (mod p)."""
        if len(x) != self.length:
            raise ValueError("x length does not match setup length.")

        p, g = int(self.p), int(self.g)
        r = random.randrange(1, p - 1)
        ct0 = pow(g, r, p)
        ct = [(pow(int(h_i), r, p) * pow(g, int(x_i) % (p - 1), p)) % p for h_i, x_i in zip(self.mpk, x)]
        return ct0, ct

    def key_derive(self, y: Sequence[int]) -> int:
        """Derive the functional key ``sk_y = <s, y> mod (p - 1)``."""
        if len(y) != self.length:
            raise ValueError("y length does not match setup length.")
        return sum((si * yi) % (self.p - 1) for si, yi in zip(self.msk, y)) % (self.p - 1)

    def decrypt(self, ct: Ciphertext, sk_y: int, y: Sequence[int], max_ip: int | None = None) -> int:
        """Recover the signed inner product ``<x, y>`` from ``ct`` using ``sk_y``.

        Args:
            max_ip: upper bound on the discrete log searched by BSGS (defaults to ``p - 1``).
        """
        if max_ip is None:
            max_ip = self.p - 1

        ct0, cts = ct
        val = decrypt_fast(ct0, cts, sk_y, y, self.p)

        ip = bsgs_numba(self.g, val, self.p, max_ip)
        if ip == -1:
            raise ValueError("Discrete log failed; increase prime or reduce message range.")

        modulus = self.p - 1
        return ip - modulus if ip > modulus // 2 else ip

    def run(self, x, y, bias, scale=1, image=False):
        """Encrypt ``x``, decrypt ``<x, y> + bias`` with float weights ``y`` and print a comparison."""
        ct = self.encrypt(x)

        if image:
            import matplotlib.pyplot as plt

            plt.imshow(np.asarray(ct[1]).reshape(28, 28), cmap="gray")
            plt.title("Encrypted image")
            plt.show()
            plt.close()

        scaled_y_input = [int(val * scale) for val in y]

        sk_y = self.key_derive(scaled_y_input)
        ip = self.decrypt(ct, sk_y, scaled_y_input) / scale

        print("p:", self.p, "g:", self.g)
        print("x:", x)
        print("y:", y)
        print("scaled y:", scaled_y_input)
        print("<x, y> (expected):", sum(xi * yi for xi, yi in zip(x, y)) + bias)
        print("<x, y> (decrypted):", ip + bias)
