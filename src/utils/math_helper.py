"""Number-theory helpers for IPFE: modular arithmetic, generators, primes and discrete logs."""

from __future__ import annotations

import math
from math import isqrt

from numba import njit


@njit
def mod_pow_numba(a, b, p):
    """Compute ``(a ** b) % p`` using binary exponentiation (Numba-friendly)."""
    result = 1
    a = a % p
    while b > 0:
        if b & 1:
            result = (result * a) % p
        a = (a * a) % p
        b >>= 1
    return result


@njit
def mod_inv_numba(a, p):
    """Modular inverse via Fermat's little theorem (``p`` must be prime)."""
    return mod_pow_numba(a, p - 2, p)


@njit
def bsgs_numba(g, h, p, max_range):
    """Solve ``g^x = h (mod p)`` for ``x`` in ``[0, max_range)`` with baby-step giant-step.

    Returns ``-1`` if no solution exists in that range.
    """
    m = int(math.ceil(math.sqrt(max_range)))
    table = {}
    e = 1
    for j in range(m):
        table[e] = j
        e = (e * g) % p
    factor = mod_pow_numba(g, m * (p - 2), p)  # g^-m mod p
    gamma = h
    for i in range(m):
        if gamma in table:
            return i * m + table[gamma]
        gamma = (gamma * factor) % p
    return -1


def inv_mod(a: int, p: int) -> int:
    """Modular inverse via Fermat's little theorem (``p`` must be prime)."""
    return pow(a, p - 2, p)


def factor(n: int) -> list[int]:
    """Return the distinct prime factors of ``n`` (trial division)."""
    f: dict[int, int] = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        f[n] = 1
    return list(f.keys())


def find_generator(p: int) -> int:
    """Return the smallest generator of the multiplicative group ``Z_p^*`` (``p`` prime)."""
    phi = p - 1
    pf = factor(phi)
    for g in range(2, p):
        if all(pow(g, phi // q, p) != 1 for q in pf):
            return g
    raise RuntimeError("no generator found")


def is_prime(n: int) -> bool:
    """Deterministic primality test by trial division (fine for the ~10^7 primes used here)."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    return all(n % d for d in range(3, isqrt(n) + 1, 2))


def next_prime(n: int | float) -> int:
    """Return the smallest prime strictly greater than ``n``."""
    candidate = math.floor(n) + 1
    while not is_prime(candidate):
        candidate += 1
    return candidate


def bsgs(g: int, h: int, p: int) -> int | None:
    """Solve ``g^x = h (mod p)`` for ``x`` in ``[0, p-2]``; returns ``None`` if not found."""
    m = isqrt(p) + 1
    # baby steps: g^j
    table = {}
    e = 1
    for j in range(m):
        table[e] = j
        e = (e * g) % p
    # giant step factor: g^{-m}
    gm = pow(g, m * (p - 2), p)
    y = h
    for i in range(m + 1):
        if y in table:
            return i * m + table[y]
        y = (y * gm) % p
    return None
