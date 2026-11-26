"""Reference (pure Python) DDH-based Inner-Product Functional Encryption (IPFE).

Basic implementation used to compute the first convolutional layer of a CNN on
encrypted image patches. See :mod:`src.cryptography.optimized_cnn_ipfe` for the
Numba-accelerated version used by the optimized models.

Reference: Abdalla, Bourse, De Caro, Pointcheval,
"Simple Functional Encryption Schemes for Inner Products", PKC 2015 (scheme on page 8).
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from src.utils.math_helper import bsgs, find_generator, inv_mod

Ciphertext = tuple[int, list[int]]


class IPFE:
    """DDH-based IPFE over the multiplicative group ``Z_p^*``.

    ``p`` must be prime and larger than twice the largest absolute inner product,
    so that signed results can be recovered from the discrete log.
    """

    def __init__(self, p: int):
        self.p = p
        self.g: int | None = None
        self.length: int | None = None
        self.mpk: list[int] | None = None
        self.msk: list[int] | None = None

    def setup(self, l: int) -> None:  # noqa: E741 - `l` is the vector length in the paper
        """Generate master keys: ``msk = s = (s_1, ..., s_l)`` and ``mpk = (h_i = g^{s_i})``."""
        self.length = l
        self.g = find_generator(self.p)
        s = [random.randrange(1, self.p - 1) for _ in range(self.length)]
        self.mpk = [pow(self.g, s_i, self.p) for s_i in s]
        self.msk = s

    def encrypt(self, x: Sequence[int]) -> Ciphertext:
        """Encrypt ``x``: ``ct0 = g^r`` and ``ct_i = h_i^r * g^{x_i}`` (mod p)."""
        if len(x) != self.length:
            raise ValueError("x length does not match setup length.")

        r = random.randrange(1, self.p - 1)
        ct0 = pow(self.g, r, self.p)
        ct = [(pow(h_i, r, self.p) * pow(self.g, x_i % (self.p - 1), self.p)) % self.p for h_i, x_i in zip(self.mpk, x)]
        return ct0, ct

    def key_derive(self, y: Sequence[int]) -> int:
        """Derive the functional key ``sk_y = <s, y> mod (p - 1)``."""
        if len(y) != self.length:
            raise ValueError("y length does not match setup length.")
        return sum((si * yi) % (self.p - 1) for si, yi in zip(self.msk, y)) % (self.p - 1)

    def decrypt(self, ct: Ciphertext, sk_y: int, y: Sequence[int]) -> int:
        """Recover the signed inner product ``<x, y>``.

        Computes ``g^<x,y> = prod_i ct_i^{y_i} / ct0^{sk_y}`` and solves the discrete log
        with baby-step giant-step, which is only feasible for small message ranges.
        """
        ct0, cts = ct

        num = 1
        for ci, yi in zip(cts, y):
            num = (num * pow(ci, yi % (self.p - 1), self.p)) % self.p
        denom = pow(ct0, sk_y % (self.p - 1), self.p)
        val = (num * inv_mod(denom, self.p)) % self.p

        ip = bsgs(self.g, val, self.p)
        if ip is None:
            raise ValueError("discrete log failed (increase prime or reduce message range).")

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


if __name__ == "__main__":
    # Example: one 3x3 MNIST patch against one trained 3x3 kernel (expected ≈ -67.167).
    p_input = 1721257
    x_input = [0, 133, 254, 9, 205, 248, 126, 254, 182]
    y_input = [-0.0300, -0.0261, -0.0194, 0.0786, 0.3495, -0.0135, -0.3420, -0.1810, -0.2126]

    ipfe_demo = IPFE(p_input)
    ipfe_demo.setup(len(x_input))
    ipfe_demo.run(x_input, y_input, bias=-0.016008036211133003, scale=10000)
