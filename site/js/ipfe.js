// DDH-based Inner-Product Functional Encryption (Abdalla et al., PKC 2015) in plain JavaScript.
// Mirrors src/cryptography/optimized_cnn_ipfe.py. All values stay below p^2 < 2^53, so ordinary
// Numbers are exact as long as p < 9.4e7 (the demo uses p ≈ 8e6).

const MAX_SAFE_PRIME = 94_906_265;

export function mod(a, m) {
  const r = a % m;
  return r < 0 ? r + m : r;
}

export function modPow(base, exp, m) {
  let result = 1;
  base = mod(base, m);
  while (exp > 0) {
    if (exp % 2 === 1) result = (result * base) % m;
    base = (base * base) % m;
    exp = Math.floor(exp / 2);
  }
  return result;
}

function primeFactors(n) {
  const factors = [];
  for (let d = 2; d * d <= n; d += d === 2 ? 1 : 2) {
    if (n % d === 0) {
      factors.push(d);
      while (n % d === 0) n /= d;
    }
  }
  if (n > 1) factors.push(n);
  return factors;
}

/** Smallest generator of Z_p^* (p prime), same as find_generator in math_helper.py. */
export function findGenerator(p) {
  const phi = p - 1;
  const qs = primeFactors(phi);
  for (let g = 2; g < p; g++) {
    if (qs.every((q) => modPow(g, phi / q, p) !== 1)) return g;
  }
  throw new Error("no generator found");
}

/** Uniform integer in [lo, hi) from the platform CSPRNG. */
export function randomInt(lo, hi) {
  const range = hi - lo;
  const buf = new Uint32Array(2);
  crypto.getRandomValues(buf);
  const r = (buf[0] * 2 ** 21 + (buf[1] >>> 11)) % range; // 53 random bits
  return lo + r;
}

export class IPFE {
  constructor(p) {
    if (p > MAX_SAFE_PRIME) throw new Error("prime too large for exact Number arithmetic");
    this.p = p;
    this.g = findGenerator(p);
    this._bsgs = null;
  }

  /** Master keys for vectors of length l: msk = s, mpk = (h_i = g^s_i). */
  setup(l) {
    this.length = l;
    this.msk = Array.from({ length: l }, () => randomInt(1, this.p - 1));
    this.mpk = this.msk.map((s) => modPow(this.g, s, this.p));
  }

  /** ct0 = g^r, ct_i = h_i^r * g^x_i (mod p). */
  encrypt(x) {
    if (x.length !== this.length) throw new Error("x length does not match setup length");
    const { p, g } = this;
    const r = randomInt(1, p - 1);
    const ct0 = modPow(g, r, p);
    const cts = x.map((xi, i) => (modPow(this.mpk[i], r, p) * modPow(g, mod(xi, p - 1), p)) % p);
    return { ct0, cts };
  }

  /** sk_y = <s, y> mod (p - 1). */
  keyDerive(y) {
    if (y.length !== this.length) throw new Error("y length does not match setup length");
    const q = this.p - 1;
    let sk = 0;
    for (let i = 0; i < y.length; i++) sk = (sk + ((this.msk[i] * mod(y[i], q)) % q)) % q;
    return sk;
  }

  /** Returns the signed inner product <x, y>; `steps` counts giant steps for the UI. */
  decrypt({ ct0, cts }, sk, y) {
    const { p } = this;
    let num = 1;
    for (let i = 0; i < cts.length; i++) num = (num * modPow(cts[i], mod(y[i], p - 1), p)) % p;
    const denom = modPow(ct0, sk, p);
    const val = (num * modPow(denom, p - 2, p)) % p; // g^<x,y>
    const ip = this.discreteLog(val);
    const q = p - 1;
    return ip > q / 2 ? ip - q : ip;
  }

  /** Baby-step giant-step: solve g^e = h (mod p) for e in [0, p-1). */
  discreteLog(h) {
    const { p, g } = this;
    if (!this._bsgs) {
      const m = Math.ceil(Math.sqrt(p - 1));
      const table = new Map();
      let e = 1;
      for (let j = 0; j < m; j++) {
        if (!table.has(e)) table.set(e, j);
        e = (e * g) % p;
      }
      this._bsgs = { m, table, factor: modPow(g, (p - 1 - m) % (p - 1), p) }; // g^-m
    }
    const { m, table, factor } = this._bsgs;
    let gamma = h;
    for (let i = 0; i < m; i++) {
      const j = table.get(gamma);
      if (j !== undefined) return i * m + j;
      gamma = (gamma * factor) % p;
    }
    throw new Error("discrete log failed: increase the prime or reduce the message range");
  }
}
