// Checks the browser implementation against PyTorch: node site/tests/check.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { argmax } from "../js/cnn.js";
import { IPFE, randomInt } from "../js/ipfe.js";
import { clientEncrypt, clientSetup, plaintextInfer, serverInfer } from "../js/pipeline.js";

const model = JSON.parse(readFileSync(new URL("../model.json", import.meta.url), "utf-8"));

// 1. IPFE round trip on random vectors.
const ipfe = new IPFE(model.ipfe.prime);
ipfe.setup(9);
for (let t = 0; t < 200; t++) {
  const x = Array.from({ length: 9 }, () => randomInt(0, 256));
  const y = Array.from({ length: 9 }, () => randomInt(-1500, 1501));
  const expected = x.reduce((s, xi, i) => s + xi * y[i], 0);
  assert.equal(ipfe.decrypt(ipfe.encrypt(x), ipfe.keyDerive(y), y), expected);
}
console.log("ok  IPFE decrypt(encrypt(x), sk_y) == <x, y> for 200 random vectors");

// 2. Plaintext forward pass matches PyTorch, and 3. encrypted inference matches plaintext.
const { ipfe: clientIpfe, ys, keys } = clientSetup(model);
let maxPlain = 0;
let maxEnc = 0;
model.samples.forEach((sample, i) => {
  const ref = model.referenceLogits[i];
  const plain = plaintextInfer(model, sample.pixels);
  maxPlain = Math.max(maxPlain, ...plain.logits.map((v, j) => Math.abs(v - ref[j])));
  assert.equal(plain.prediction, argmax(ref), `sample ${i}: plaintext prediction differs from PyTorch`);

  const { ciphertexts, out } = clientEncrypt(model, clientIpfe, sample.pixels);
  const enc = serverInfer(model, clientIpfe, ys, keys, ciphertexts, out);
  maxEnc = Math.max(maxEnc, ...enc.logits.map((v, j) => Math.abs(v - plain.logits[j])));
  assert.equal(enc.prediction, plain.prediction, `sample ${i}: encrypted prediction differs`);
});
assert.ok(maxPlain < 1e-3, `plaintext logits differ from PyTorch by ${maxPlain}`);
// Weights are rounded to 1e-4 for the integer keys, so first-layer outputs differ by up to ~0.1.
assert.ok(maxEnc < 0.5, `encrypted logits differ from plaintext by ${maxEnc}`);
console.log(`ok  ${model.samples.length} samples: plaintext vs PyTorch max |Δlogit| = ${maxPlain.toExponential(2)}`);
console.log(`ok  ${model.samples.length} samples: encrypted vs plaintext max |Δlogit| = ${maxEnc.toExponential(2)}`);
