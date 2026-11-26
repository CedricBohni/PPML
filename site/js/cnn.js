// Minimal CNN inference for the exported PPML model (see scripts/export_web_model.py).
// Tensors are Float64Arrays in (channels, height, width) order.

/** Split a 28x28 image into k*k patches exactly like torch.nn.Unfold (zero padding). */
export function unfold(pixels, size, k, stride, pad) {
  const out = Math.floor((size + 2 * pad - k) / stride) + 1;
  const patches = [];
  for (let oy = 0; oy < out; oy++) {
    for (let ox = 0; ox < out; ox++) {
      const patch = [];
      for (let ky = 0; ky < k; ky++) {
        for (let kx = 0; kx < k; kx++) {
          const y = oy * stride + ky - pad;
          const x = ox * stride + kx - pad;
          patch.push(y >= 0 && y < size && x >= 0 && x < size ? pixels[y * size + x] : 0);
        }
      }
      patches.push(patch);
    }
  }
  return { patches, out };
}

function conv2d(input, C, H, W, layer) {
  const weight = layer.weight; // [O][C][k][k]
  const O = weight.length;
  const k = weight[0][0].length;
  const { stride, padding } = layer;
  const Ho = Math.floor((H + 2 * padding - k) / stride) + 1;
  const Wo = Math.floor((W + 2 * padding - k) / stride) + 1;
  const out = new Float64Array(O * Ho * Wo);
  for (let o = 0; o < O; o++) {
    for (let oy = 0; oy < Ho; oy++) {
      for (let ox = 0; ox < Wo; ox++) {
        let acc = layer.bias[o];
        for (let c = 0; c < C; c++) {
          for (let ky = 0; ky < k; ky++) {
            const y = oy * stride + ky - padding;
            if (y < 0 || y >= H) continue;
            for (let kx = 0; kx < k; kx++) {
              const x = ox * stride + kx - padding;
              if (x < 0 || x >= W) continue;
              acc += weight[o][c][ky][kx] * input[(c * H + y) * W + x];
            }
          }
        }
        out[(o * Ho + oy) * Wo + ox] = acc;
      }
    }
  }
  return { data: out, C: O, H: Ho, W: Wo };
}

function bnRelu(t, layer) {
  const n = t.H * t.W;
  for (let c = 0; c < t.C; c++) {
    const s = layer.bnScale[c];
    const b = layer.bnShift[c];
    for (let i = c * n; i < (c + 1) * n; i++) t.data[i] = Math.max(0, t.data[i] * s + b);
  }
  return t;
}

function maxPool2(t) {
  const Ho = Math.floor(t.H / 2);
  const Wo = Math.floor(t.W / 2);
  const out = new Float64Array(t.C * Ho * Wo);
  for (let c = 0; c < t.C; c++) {
    for (let y = 0; y < Ho; y++) {
      for (let x = 0; x < Wo; x++) {
        let m = -Infinity;
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) m = Math.max(m, t.data[(c * t.H + 2 * y + dy) * t.W + 2 * x + dx]);
        }
        out[(c * Ho + y) * Wo + x] = m;
      }
    }
  }
  return { data: out, C: t.C, H: Ho, W: Wo };
}

function linear(input, layer, relu) {
  return layer.weight.map((row, o) => {
    let acc = layer.bias[o];
    for (let i = 0; i < row.length; i++) acc += row[i] * input[i];
    return relu ? Math.max(0, acc) : acc;
  });
}

/** First convolution in plaintext: returns {data, C, H, W} before batch norm. */
export function firstConv(model, pixels) {
  return conv2d(Float64Array.from(pixels), 1, 28, 28, model.conv[0]);
}

/** Everything after the first convolution: BN/ReLU/pool of layer 1, conv layers 2..n, FC layers. */
export function forwardFromFirstConv(model, conv1) {
  let t = { data: Float64Array.from(conv1.data), C: conv1.C, H: conv1.H, W: conv1.W };
  model.conv.forEach((layer, i) => {
    if (i > 0) t = conv2d(t.data, t.C, t.H, t.W, layer);
    t = bnRelu(t, layer);
    if (layer.pool) t = maxPool2(t);
  });
  const hidden = linear(t.data, model.fc[0], true); // dropout is a no-op at inference
  return linear(hidden, model.fc[1], false);
}

export function softmax(logits) {
  const m = Math.max(...logits);
  const e = logits.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export const argmax = (v) => v.reduce((best, x, i) => (x > v[best] ? i : best), 0);

/** Integer kernel vectors y (weights scaled and rounded) used for IPFE key derivation. */
export function kernelVectors(model) {
  const scale = model.ipfe.weightScale;
  return model.conv[0].weight.map((w) => w.flat(2).map((v) => Math.round(v * scale)));
}
