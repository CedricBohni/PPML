import random

import pytest

from src.utils.math_helper import (
    bsgs,
    bsgs_numba,
    factor,
    find_generator,
    inv_mod,
    is_prime,
    mod_inv_numba,
    mod_pow_numba,
    next_prime,
)

PRIMES = [101, 104729, 1721257, 6285761]


@pytest.mark.parametrize("p", PRIMES)
def test_mod_pow_matches_builtin(p):
    rng = random.Random(p)
    for _ in range(50):
        a, b = rng.randrange(p), rng.randrange(p)
        assert mod_pow_numba(a, b, p) == pow(a, b, p)


@pytest.mark.parametrize("p", PRIMES)
def test_modular_inverse(p):
    for a in (1, 2, 3, p - 1, p // 2):
        assert (a * inv_mod(a, p)) % p == 1
        assert (a * mod_inv_numba(a, p)) % p == 1


def test_factor_returns_distinct_prime_factors():
    assert sorted(factor(2**4 * 3 * 7**2)) == [2, 3, 7]
    assert factor(97) == [97]


@pytest.mark.parametrize("p", PRIMES)
def test_find_generator_generates_whole_group(p):
    g = find_generator(p)
    # g is a generator iff g^((p-1)/q) != 1 for every prime factor q of p-1.
    assert all(pow(g, (p - 1) // q, p) != 1 for q in factor(p - 1))
    if p < 2000:
        assert len({pow(g, k, p) for k in range(p - 1)}) == p - 1


@pytest.mark.parametrize("p", [104729, 1721257])
def test_discrete_log_solvers_agree(p):
    g = find_generator(p)
    rng = random.Random(0)
    for x in [0, 1, 2, p - 2] + [rng.randrange(p - 1) for _ in range(20)]:
        h = pow(g, x, p)
        assert bsgs(g, h, p) == x
        assert bsgs_numba(g, h, p, p - 1) == x


def test_bsgs_numba_out_of_range_returns_minus_one():
    p = 104729
    g = find_generator(p)
    assert bsgs_numba(g, pow(g, 5000, p), p, 100) == -1


def test_primes():
    assert [n for n in range(30) if is_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert is_prime(6285761)
    assert next_prime(10) == 11
    assert next_prime(11) == 13
    assert next_prime(6266634.5) > 6266634.5 and is_prime(next_prime(6266634.5))
