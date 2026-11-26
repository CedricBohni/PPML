// The encrypted-inference protocol, split into the client and server roles.
import { argmax, firstConv, forwardFromFirstConv, kernelVectors, softmax, unfold } from "./cnn.js";
import { IPFE } from "./ipfe.js";

/** Client: generate master keys and derive one functional key per first-layer kernel. */
export function clientSetup(model) {
  const ipfe = new IPFE(model.ipfe.prime);
  const k = model.conv[0].weight[0][0].length;
  ipfe.setup(k * k);
  const ys = kernelVectors(model); // in the demo protocol the server shares its kernels
  const keys = ys.map((y) => ipfe.keyDerive(y));
  return { ipfe, ys, keys };
}

/** Client: unfold the image into patches and encrypt each one. */
export function clientEncrypt(model, ipfe, pixels) {
  const { stride, padding } = model.conv[0];
  const k = model.conv[0].weight[0][0].length;
  const { patches, out } = unfold(pixels, 28, k, stride, padding);
  const t0 = performance.now();
  const ciphertexts = patches.map((x) => ipfe.encrypt(x));
  return { patches, out, ciphertexts, ms: performance.now() - t0 };
}

/**
 * Server: decrypt <x, y> for every patch and kernel, then finish the CNN in plaintext.
 * The server only needs p, g, the keys sk_y and its own kernels y, never the master key.
 */
export function serverInfer(model, ipfe, ys, keys, ciphertexts, out) {
  const scale = model.ipfe.weightScale;
  const bias = model.conv[0].bias;
  const C = ys.length;
  const data = new Float64Array(C * out * out);
  const t0 = performance.now();
  for (let c = 0; c < C; c++) {
    for (let p = 0; p < ciphertexts.length; p++) {
      data[c * out * out + p] = ipfe.decrypt(ciphertexts[p], keys[c], ys[c]) / scale + bias[c];
    }
  }
  const decryptMs = performance.now() - t0;
  const conv1 = { data, C, H: out, W: out };
  const logits = forwardFromFirstConv(model, conv1);
  return { conv1, logits, probs: softmax(logits), prediction: argmax(logits), decryptMs };
}

/** Reference: the same network entirely in plaintext. */
export function plaintextInfer(model, pixels) {
  const conv1 = firstConv(model, pixels);
  const logits = forwardFromFirstConv(model, conv1);
  return { conv1, logits, probs: softmax(logits), prediction: argmax(logits) };
}
