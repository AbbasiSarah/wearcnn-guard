// WearCNN-Guard in the browser: same preprocessing, CNN and conformal step as the Python code.
// Everything runs locally; no data leaves the device.
(function (root) {
  "use strict";

  // ---------- 1. read a CSV of minute-level activity ----------
  function parseCSV(text) {
    const lines = text.replace(/\r/g, "").split("\n").filter(l => l.trim().length);
    if (lines.length < 2) throw new Error("The file looks empty.");
    const sep = lines[0].includes(";") && !lines[0].includes(",") ? ";" : ",";
    const head = lines[0].split(sep).map(h => h.trim().toLowerCase().replace(/"/g, ""));
    let ti = head.findIndex(h => /time|date/.test(h));
    let ai = head.findIndex(h => /activity|count|value|acc|movement/.test(h));
    if (ti < 0 || ai < 0) throw new Error('Need two columns: "timestamp" and "activity".');
    const days = new Map();
    let n = 0;
    for (let i = 1; i < lines.length; i++) {
      const c = lines[i].split(sep);
      if (c.length <= Math.max(ti, ai)) continue;
      const ts = c[ti].trim().replace(/"/g, "");
      const m = ts.match(/(\d{4})-(\d{1,2})-(\d{1,2})[ T](\d{1,2}):(\d{2})/);
      if (!m) continue;
      const v = parseFloat(c[ai]);
      if (!isFinite(v) || v < 0) continue;
      const key = `${m[1]}-${m[2].padStart(2, "0")}-${m[3].padStart(2, "0")}`;
      const minute = (+m[4]) * 60 + (+m[5]);
      if (!days.has(key)) days.set(key, new Float64Array(1440).fill(NaN));
      days.get(key)[minute] = v; n++;
    }
    if (!n) throw new Error('No rows could be read. Timestamps should look like "2026-01-05 13:45".');
    return [...days.keys()].sort().map(k => ({ date: k, x: days.get(k) }));
  }

  // ---------- 2. turn days into the model input (same rules as prep.py) ----------
  function dayProfile(x) {
    const z = new Uint8Array(1440), nonwear = new Uint8Array(1440), run = new Int32Array(1440);
    for (let j = 0; j < 1440; j++) {
      z[j] = isFinite(x[j]) && x[j] === 0 ? 1 : 0;
      run[j] = z[j] ? (j ? run[j - 1] + 1 : 1) : 0;
    }
    let endLen = 0;
    for (let j = 1439; j >= 0; j--) {
      endLen = z[j] ? Math.max(endLen, run[j]) : 0;
      nonwear[j] = z[j] && endLen >= 60 ? 1 : 0;
    }
    let wear = 0;
    const prof = new Float64Array(144);
    for (let b = 0; b < 144; b++) {
      let s = 0, k = 0;
      for (let j = b * 10; j < b * 10 + 10; j++) {
        if (isFinite(x[j]) && !nonwear[j]) { s += Math.log1p(x[j]); k++; wear++; }
      }
      prof[b] = k ? s / k : NaN;
    }
    return { wear, prof };
  }

  function buildInput(days) {
    const valid = [];
    const info = days.map(d => { const p = dayProfile(d.x); const ok = p.wear >= 600; if (ok && valid.length < 7) valid.push(p.prof); return { date: d.date, wearHours: p.wear / 60, valid: ok }; });
    if (valid.length < 4) return { ok: false, info, nValid: valid.length };
    const meanDay = new Float64Array(144);
    for (let b = 0; b < 144; b++) {
      let s = 0, k = 0; for (const p of valid) if (isFinite(p[b])) { s += p[b]; k++; }
      meanDay[b] = k ? s / k : NaN;
    }
    let s = 0, k = 0; for (const v of meanDay) if (isFinite(v)) { s += v; k++; }
    const overall = s / k;
    for (let b = 0; b < 144; b++) if (!isFinite(meanDay[b])) meanDay[b] = overall;
    const P = valid.map(p => p.map((v, b) => isFinite(v) ? v : meanDay[b]));
    const mean = new Float64Array(144), sd = new Float64Array(144);
    for (let b = 0; b < 144; b++) {
      let m = 0; for (const p of P) m += p[b]; m /= P.length; mean[b] = m;
      let q = 0; for (const p of P) q += (p[b] - m) ** 2; sd[b] = Math.sqrt(q / P.length);
    }
    return { ok: true, info, nValid: valid.length, input: [Array.from(mean), Array.from(sd)] };
  }

  // ---------- 3. the CNN forward pass ----------
  function conv(x, L) {           // x: [C][len], W: [(C*k)][out] with row = c*k + t
    const C = x.length, len = x[0].length, k = L.k, out = L.b.length, left = Math.floor((k - 1) / 2);
    const y = Array.from({ length: out }, () => new Float64Array(len));
    for (let i = 0; i < len; i++) {
      for (let o = 0; o < out; o++) {
        let s = L.b[o];
        for (let c = 0; c < C; c++) for (let t = 0; t < k; t++) {
          const p = i + t - left; if (p < 0 || p >= len) continue;
          s += x[c][p] * L.W[c * k + t][o];
        }
        y[o][i] = s;
      }
    }
    return y;
  }
  const relu = x => x.map(r => r.map(v => (v > 0 ? v : 0)));
  function maxpool(x, size) {
    return x.map(r => { const n = Math.floor(r.length / size), o = new Float64Array(n); for (let i = 0; i < n; i++) { let m = -Infinity; for (let j = 0; j < size; j++) m = Math.max(m, r[i * size + j]); o[i] = m; } return o; });
  }
  function forward(model, input) {
    let x = input.map((r, c) => r.map(v => (v - model.mu[c]) / model.sd[c]));
    let vec = null;
    for (const L of model.layers) {
      if (L.type === "conv") x = conv(x, L);
      else if (L.type === "relu") x = relu(x);
      else if (L.type === "maxpool") x = maxpool(x, L.size);
      else if (L.type === "gap") vec = x.map(r => r.reduce((a, b) => a + b, 0) / r.length);
      else if (L.type === "dense") { let s = L.b[0]; for (let i = 0; i < vec.length; i++) s += vec[i] * L.W[i][0]; return 1 / (1 + Math.exp(-s)); }
    }
  }

  // ---------- 4. conformal answer ----------
  function predict(bundle, input) {
    const p = bundle.models.reduce((a, m) => a + forward(m, input), 0) / bundle.models.length;
    const inNot = p <= bundle.qhat[0];        // score for "not depressed" = p
    const inDep = 1 - p <= bundle.qhat[1];    // score for "depressed" = 1 - p
    let answer = "unsure";
    if (inDep && !inNot) answer = "depressed";
    else if (inNot && !inDep) answer = "not";
    return { p, answer, set: { not: inNot, depressed: inDep } };
  }

  const api = { parseCSV, buildInput, forward, predict };
  if (typeof module !== "undefined") module.exports = api; else root.WearCNN = api;
})(this);
