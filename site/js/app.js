import { clientEncrypt, clientSetup, plaintextInfer, serverInfer } from "./pipeline.js";

const $ = (id) => document.getElementById(id);
const GRID = 10; // patches per row (28x28, k=3, stride 3, pad 1)
const dark = matchMedia("(prefers-color-scheme: dark)");

let model;
let setup;
let state = null;
let hovered = null;

// ---------- drawing helpers ----------

function putPixels(canvas, w, h, rgbAt) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const [r, g, b] = rgbAt(i);
    img.data.set([r, g, b, 255], i * 4);
  }
  ctx.putImageData(img, 0, 0);
}

const gray = (canvas, w, h, values) => putPixels(canvas, w, h, (i) => [values[i], values[i], values[i]]);

const hex = (s) => [1, 3, 5].map((i) => parseInt(s.slice(i, i + 2), 16));
const lerp = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

/** Diverging map: negative -> blue, zero -> neutral gray, positive -> orange. */
function diverging(v, maxAbs) {
  const neg = hex(dark.matches ? "#338ced" : "#257de0");
  const pos = hex(dark.matches ? "#e76624" : "#f47230");
  const mid = hex(dark.matches ? "#3a3a3e" : "#e6e6e3");
  const t = maxAbs > 0 ? Math.max(-1, Math.min(1, v / maxAbs)) : 0;
  return t < 0 ? lerp(mid, neg, -t) : lerp(mid, pos, t);
}

function fmt(n, digits = 3) {
  return Number(n).toLocaleString("en-US", { maximumFractionDigits: digits });
}

function miniTable(values, cols = 3) {
  let html = "<table class='mini'>";
  for (let r = 0; r < values.length / cols; r++) {
    html += "<tr>" + values.slice(r * cols, r * cols + cols).map((v) => `<td>${v}</td>`).join("") + "</tr>";
  }
  return html + "</table>";
}

// ---------- step 1: inputs ----------

function renderSamples() {
  const box = $("samples");
  model.samples.forEach((s, i) => {
    const btn = document.createElement("button");
    btn.className = "sample";
    btn.setAttribute("role", "option");
    btn.setAttribute("aria-label", `Test image ${i + 1}, label ${s.label}`);
    const c = document.createElement("canvas");
    gray(c, 28, 28, s.pixels);
    btn.append(c);
    btn.addEventListener("click", () => {
      selectSample(i);
      classify(s.pixels, s.label, btn);
    });
    box.append(btn);
  });
}

function selectSample(i) {
  document.querySelectorAll(".sample").forEach((b, j) => b.setAttribute("aria-selected", String(i === j)));
}

function setupPad() {
  const pad = $("pad");
  const ctx = pad.getContext("2d");
  const clear = () => {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, pad.width, pad.height);
  };
  clear();
  let drawing = false;
  let last = null;
  const pos = (e) => {
    const r = pad.getBoundingClientRect();
    return [((e.clientX - r.left) * pad.width) / r.width, ((e.clientY - r.top) * pad.height) / r.height];
  };
  pad.addEventListener("pointerdown", (e) => {
    drawing = true;
    last = pos(e);
    pad.setPointerCapture(e.pointerId);
  });
  pad.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    const p = pos(e);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 22;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(...last);
    ctx.lineTo(...p);
    ctx.stroke();
    last = p;
  });
  const stop = () => (drawing = false);
  pad.addEventListener("pointerup", stop);
  pad.addEventListener("pointercancel", stop);
  pad.addEventListener("pointerdown", () => ($("pad-hint").hidden = true));
  $("pad-clear").addEventListener("click", clear);
  $("pad-use").addEventListener("click", () => {
    const pixels = padToMnist(pad);
    if (!pixels) {
      $("pad-hint").hidden = false;
      return;
    }
    selectSample(-1);
    classify(pixels, null, $("pad-use"));
  });
}

/** MNIST-style preprocessing: crop, fit into 20x20, centre by mass in a 28x28 frame. */
function padToMnist(pad) {
  const src = pad.getContext("2d").getImageData(0, 0, pad.width, pad.height).data;
  let x0 = pad.width, y0 = pad.height, x1 = -1, y1 = -1;
  for (let y = 0; y < pad.height; y++) {
    for (let x = 0; x < pad.width; x++) {
      if (src[(y * pad.width + x) * 4] > 30) {
        x0 = Math.min(x0, x); x1 = Math.max(x1, x);
        y0 = Math.min(y0, y); y1 = Math.max(y1, y);
      }
    }
  }
  if (x1 < 0) return null;
  const w = x1 - x0 + 1, h = y1 - y0 + 1;
  const s = 20 / Math.max(w, h);
  const tw = Math.max(1, Math.round(w * s)), th = Math.max(1, Math.round(h * s));
  const small = document.createElement("canvas");
  small.width = tw;
  small.height = th;
  const sctx = small.getContext("2d");
  sctx.imageSmoothingQuality = "high";
  sctx.drawImage(pad, x0, y0, w, h, 0, 0, tw, th);
  const sd = sctx.getImageData(0, 0, tw, th).data;
  let mass = 0, cx = 0, cy = 0;
  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) {
      const v = sd[(y * tw + x) * 4];
      mass += v; cx += v * x; cy += v * y;
    }
  }
  const ox = Math.round(14 - cx / mass), oy = Math.round(14 - cy / mass);
  const out = new Array(784).fill(0);
  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) {
      const X = x + ox, Y = y + oy;
      if (X >= 0 && X < 28 && Y >= 0 && Y < 28) out[Y * 28 + X] = sd[(y * tw + x) * 4];
    }
  }
  return out;
}

// ---------- step 2: keys ----------

function renderKeys() {
  const { ipfe, ys, keys } = setup;
  $("key-facts").innerHTML = `
    <dt>prime <var>p</var></dt><dd><code>${fmt(ipfe.p)}</code></dd>
    <dt>generator <var>g</var></dt><dd><code>${ipfe.g}</code></dd>
    <dt>vector length</dt><dd><code>${ipfe.length}</code> (3×3 patch)</dd>
    <dt>public key <var>h</var></dt><dd><code>${ipfe.mpk.join(", ")}</code></dd>
    <dt>master key <var>s</var></dt><dd>9 random secrets <span class="muted">(never leave the client)</span></dd>`;
  const box = $("kernels");
  box.innerHTML = "";
  const maxAbs = Math.max(...ys.flat().map(Math.abs));
  ys.forEach((y, k) => {
    const div = document.createElement("div");
    div.className = "kernel";
    const c = document.createElement("canvas");
    putPixels(c, 3, 3, (i) => diverging(y[i], maxAbs));
    c.title = `y = [${y.join(", ")}]`;
    div.append(c, `kernel ${k + 1}`, document.createElement("br"));
    const sk = document.createElement("code");
    sk.textContent = `sk = ${fmt(keys[k])}`;
    div.append(sk);
    box.append(div);
  });
}

// ---------- step 3: encryption ----------

function patchLayout(values) {
  // values[p][j] -> 30x30 image where patch p occupies a 3x3 cell
  const out = new Array(900);
  values.forEach((patch, p) => {
    const oy = Math.floor(p / GRID), ox = p % GRID;
    patch.forEach((v, j) => {
      out[(oy * 3 + Math.floor(j / 3)) * 30 + ox * 3 + (j % 3)] = v;
    });
  });
  return out;
}

function renderEncryption() {
  const { pixels, enc } = state;
  gray($("cv-input"), 28, 28, pixels);
  gray($("cv-patches"), 30, 30, patchLayout(enc.patches));
  const p = setup.ipfe.p;
  gray($("cv-cipher"), 30, 30, patchLayout(enc.ciphertexts.map((c) => c.cts.map((v) => Math.floor((v / p) * 256)))));
  $("enc-stats").textContent = `100 patches encrypted in ${fmt(enc.ms, 1)} ms`;
  drawOverlays();
  renderInspector();
}

function drawOverlays() {
  for (const id of ["ov-patches", "ov-cipher"]) {
    const c = $(id);
    const ctx = c.getContext("2d");
    const cell = c.width / GRID;
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.strokeStyle = "rgba(128,128,128,0.35)";
    ctx.lineWidth = 1;
    for (let i = 1; i < GRID; i++) {
      ctx.beginPath(); ctx.moveTo(i * cell + 0.5, 0); ctx.lineTo(i * cell + 0.5, c.height); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * cell + 0.5); ctx.lineTo(c.width, i * cell + 0.5); ctx.stroke();
    }
    if (hovered !== null) {
      ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue("--accent");
      ctx.lineWidth = 3;
      ctx.strokeRect((hovered % GRID) * cell + 1.5, Math.floor(hovered / GRID) * cell + 1.5, cell - 3, cell - 3);
    }
  }
}

function renderInspector() {
  const box = $("inspector");
  const { enc, res } = state;
  const p = hovered ?? state.inkiest;
  const x = enc.patches[p];
  const ct = enc.ciphertexts[p];
  const scale = model.ipfe.weightScale;
  const n = GRID * GRID;
  const ips = setup.ys.map((_, k) => Math.round((res.conv1.data[k * n + p] - model.conv[0].bias[k]) * scale));
  box.innerHTML = `
    <div>
      <div class="k">patch (${Math.floor(p / GRID) + 1}, ${(p % GRID) + 1}): pixels <var>x</var></div>
      ${miniTable(x)}
    </div>
    <div>
      <div class="k">ciphertext sent to the server</div>
      <code>ct₀ = ${ct.ct0}</code><br>
      <code>ct = [${ct.cts.join(", ")}]</code>
      <div class="k" style="margin-top:8px">what the server decrypts with each key</div>
      <code>${ips.map((v, k) => `⟨x, y${k + 1}⟩ = ${v}`).join(" · ")}</code>
    </div>`;
}

function setupHover() {
  for (const id of ["ov-patches", "ov-cipher"]) {
    const c = $(id);
    c.addEventListener("pointermove", (e) => {
      const r = c.getBoundingClientRect();
      const gx = Math.min(GRID - 1, Math.floor(((e.clientX - r.left) / r.width) * GRID));
      const gy = Math.min(GRID - 1, Math.floor(((e.clientY - r.top) / r.height) * GRID));
      const p = gy * GRID + gx;
      if (p !== hovered) {
        hovered = p;
        drawOverlays();
        renderInspector();
      }
    });
  }
}

// ---------- steps 4 & 5: server ----------

function renderMaps() {
  const { res, plain } = state;
  const box = $("maps");
  box.innerHTML = "<div></div>";
  const C = res.conv1.C;
  const n = GRID * GRID;
  for (let k = 0; k < C; k++) box.insertAdjacentHTML("beforeend", `<div class="head">kernel ${k + 1}</div>`);
  let maxErr = 0;
  for (const [label, t] of [["decrypted", res.conv1], ["plaintext", plain.conv1]]) {
    box.insertAdjacentHTML("beforeend", `<div class="lab">${label}</div>`);
    for (let k = 0; k < C; k++) {
      const slice = t.data.slice(k * n, (k + 1) * n);
      const ref = plain.conv1.data.slice(k * n, (k + 1) * n);
      const maxAbs = Math.max(...ref.map(Math.abs));
      const c = document.createElement("canvas");
      putPixels(c, GRID, GRID, (i) => diverging(slice[i], maxAbs));
      box.append(c);
    }
  }
  for (let i = 0; i < res.conv1.data.length; i++) {
    maxErr = Math.max(maxErr, Math.abs(res.conv1.data[i] - plain.conv1.data[i]));
  }
  $("dec-stats").textContent =
    `${C * n} decryptions (100 patches × ${C} kernels) in ${fmt(res.decryptMs, 0)} ms. ` +
    `Largest difference from the plaintext convolution: ${maxErr.toExponential(1)}, caused by rounding the ` +
    `weights to 4 decimals for the integer keys.`;
}

function renderPrediction() {
  const { res, plain, label } = state;
  const agree = res.prediction === plain.prediction;
  let verdict = `<div class="muted">prediction from encrypted input</div><div class="digit">${res.prediction}</div>`;
  if (label !== null) {
    const ok = label === res.prediction;
    verdict += `<div class="${ok ? "ok" : "no"}">${ok ? "✓ correct" : "✗ wrong"}: true label ${label}</div>`;
  }
  verdict += `<div class="muted">plaintext model predicts ${plain.prediction}${agree ? " (same)" : " (different)"}</div>`;
  $("verdict").innerHTML = verdict;
  $("bars").innerHTML = res.probs
    .map(
      (p, d) => `<div class="bar${d === res.prediction ? " top" : ""}">
        <span>${d}</span><span class="track"><span class="fill" style="display:block;width:${(p * 100).toFixed(2)}%"></span></span>
        <span class="val">${(p * 100).toFixed(p >= 0.995 || p < 0.001 ? 0 : 1)}%</span></div>`,
    )
    .join("");
}

/** Restart a CSS animation class on an element (removing and re-adding alone does not replay it). */
function replay(el, cls) {
  el.classList.remove(cls);
  void el.offsetWidth; // force reflow
  el.classList.add(cls);
}

/** Result summary directly under the inputs, plus a highlight on every step that changed. */
function renderQuick(animate) {
  const { pixels, label, res, enc } = state;
  const box = $("quick");
  box.className = "quick";
  let verdict = "";
  if (label !== null) {
    verdict = label === res.prediction
      ? `<span class="ok">✓ correct (true label ${label})</span>`
      : `<span class="no">✗ wrong (true label ${label})</span>`;
  }
  box.innerHTML = `
    <canvas width="28" height="28" aria-label="Encrypted input"></canvas>
    <span class="qdigit">${res.prediction}</span>
    <span class="qtext">
      <strong>Predicted from encrypted patches</strong>
      <span class="muted">100 patches encrypted, ${res.conv1.C * 100} inner products decrypted in
        ${fmt(enc.ms + res.decryptMs, 0)} ms ${verdict ? "· " + verdict : ""}</span>
    </span>
    <a href="#step-encrypt" id="quick-go">See how it was computed ↓</a>`;
  gray(box.querySelector("canvas"), 28, 28, pixels);
  $("quick-go").addEventListener("click", (e) => {
    e.preventDefault();
    $("step-encrypt").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  if (animate) {
    replay(box, "pop");
    for (const id of ["step-encrypt", "step-decrypt", "step-predict"]) replay($(id), "updated");
  }
}

// ---------- pipeline ----------

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const STAGES = [
  "Encrypting 100 patches on the client…",
  "Server decrypting 800 inner products…",
  "Running the rest of the network…",
];
let busy = false;

/**
 * Click handler for samples and the drawing pad. The computation itself takes ~20 ms, so the
 * stages are paced (~1 s in total) to make it visible that the click started the protocol.
 */
async function classify(pixels, label, trigger) {
  if (busy) return;
  busy = true;
  const box = $("quick");
  const lbl = trigger.querySelector(".lbl");
  const original = lbl?.textContent;
  trigger.classList.add("loading");
  trigger.disabled = true;
  if (lbl) lbl.textContent = "Encrypting…";
  const pause = matchMedia("(prefers-reduced-motion: reduce)").matches ? 150 : 340;
  try {
    for (let i = 0; i < STAGES.length; i++) {
      box.className = "quick loading";
      box.innerHTML = `<span class="spinner" aria-hidden="true"></span>
        <span class="stage">${STAGES[i]}</span>
        <span class="steps" aria-hidden="true">${STAGES.map((_, j) => `<i class="${j <= i ? "on" : ""}"></i>`).join("")}</span>`;
      await sleep(pause);
    }
    run(pixels, label, true);
  } finally {
    trigger.classList.remove("loading");
    trigger.disabled = false;
    if (lbl) lbl.textContent = original;
    busy = false;
  }
}

function run(pixels, label, animate = false) {
  const enc = clientEncrypt(model, setup.ipfe, pixels);
  const res = serverInfer(model, setup.ipfe, setup.ys, setup.keys, enc.ciphertexts, enc.out);
  const plain = plaintextInfer(model, pixels);
  const ink = enc.patches.map((x) => x.reduce((a, b) => a + b, 0));
  const inkiest = ink.indexOf(Math.max(...ink)); // default patch for the inspector
  hovered = null;
  state = { pixels, label, enc, res, plain, inkiest };
  renderEncryption();
  renderMaps();
  renderPrediction();
  renderQuick(animate);
}

async function main() {
  model = await (await fetch("model.json")).json();
  setup = clientSetup(model);
  renderSamples();
  renderKeys();
  setupPad();
  setupHover();
  $("reencrypt").addEventListener("click", () => {
    if (!state) return;
    run(state.pixels, state.label);
    replay($("step-encrypt"), "updated");
  });
  dark.addEventListener("change", () => {
    renderKeys();
    if (state) renderMaps();
  });
  $("model-info").textContent =
    `Model: ${model.source} (first layer: 8 kernels of 3×3, stride 3). ` +
    `${model.samples.length} MNIST test images bundled.`;
  selectSample(0);
  run(model.samples[0].pixels, model.samples[0].label);
}

main().catch((err) => {
  document.querySelector("main").insertAdjacentHTML("afterbegin", `<p class="step">Failed to load the demo: ${err}</p>`);
  throw err;
});
