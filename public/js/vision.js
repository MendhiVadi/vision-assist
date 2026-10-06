/* In-browser detection: runs the YOLO ONNX models with onnxruntime-web, no server involved.
 * Port of detector.py + scene.py + the API's summarize/normalize. Exposes window.Vision.
 *
 *   await Vision.analyze(canvasOrImage, 'live' | 'lens')
 *     -> { output, objects, width, height, detections: [{label, confidence, box:[x1,y1,x2,y2]}] }
 */
(() => {
  const SIZE = 640;
  const MODELS = {
    // `wasm` is used when WebGPU is unavailable (smaller = usable speed on CPU).
    live: { url: '/models/live.onnx', wasmUrl: '/models/live-lite.onnx', labels: '/models/live.json', wasmLabels: '/models/live.json', conf: 0.45 },
    lens: { url: '/models/lens.onnx', labels: '/models/lens.json', rules: '/models/lens_rules.json', conf: 0.3 },
  };
  const IOU = 0.5, MAX_DET = 30, CLOSE = 0.25, NEAR = 0.08;

  const ALIASES = {
    'cabinet': 'cupboard', 'kitchen cabinet': 'cupboard', 'cabinetry': 'cupboard',
    'side cabinet': 'cupboard', 'bathroom cabinet': 'cupboard', 'closet': 'cupboard',
    'beer can': 'can', 'spray can': 'can', 'milk can': 'can',
    'earphone': 'earphones', 'stop watch': 'watch', 'pocket watch': 'watch',
    'power plugs and sockets': 'power socket',
  };

  // ---------- onnxruntime setup ----------
  const hasGPU = !!navigator.gpu;
  ort.env.wasm.wasmPaths = '/ort/';
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

  const sessions = {};
  async function load(kind, onStatus) {
    if (sessions[kind]) return sessions[kind];
    sessions[kind] = (async () => {
      const m = MODELS[kind];
      const useGPU = hasGPU;
      const url = useGPU || !m.wasmUrl ? m.url : m.wasmUrl;
      const labelsUrl = useGPU || !m.wasmLabels ? m.labels : m.wasmLabels;
      onStatus && onStatus(`Loading ${kind} model (first time only)...`);
      const [labels, rules, buf] = await Promise.all([
        fetch(labelsUrl).then((r) => r.json()),
        m.rules ? fetch(m.rules).then((r) => r.json()).catch(() => null) : null,
        fetch(url).then((r) => { if (!r.ok) throw new Error('model download failed'); return r.arrayBuffer(); }),
      ]);
      let session;
      if (useGPU) {
        try { session = await ort.InferenceSession.create(buf, { executionProviders: ['webgpu', 'wasm'] }); }
        catch (e) { session = null; }
      }
      if (!session) session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
      // Web-tag vocabulary: collapse "jazz artist"/"rocketer" etc. to "person", suppress non-physical tags.
      const person = new Set(rules && rules.person), drop = new Set(rules && rules.drop), merge = (rules && rules.merge) || {};
      return { session, labels, conf: m.conf, person, drop, merge };
    })();
    sessions[kind].catch(() => { delete sessions[kind]; });
    return sessions[kind];
  }

  // ---------- preprocess (letterbox, like Ultralytics) ----------
  const work = document.createElement('canvas');
  work.width = work.height = SIZE;
  const wctx = work.getContext('2d', { willReadFrequently: true });

  function preprocess(src) {
    const w = src.videoWidth || src.naturalWidth || src.width;
    const h = src.videoHeight || src.naturalHeight || src.height;
    const r = Math.min(SIZE / w, SIZE / h);
    const nw = Math.round(w * r), nh = Math.round(h * r);
    const dx = Math.floor((SIZE - nw) / 2), dy = Math.floor((SIZE - nh) / 2);
    wctx.fillStyle = 'rgb(114,114,114)';
    wctx.fillRect(0, 0, SIZE, SIZE);
    wctx.drawImage(src, 0, 0, w, h, dx, dy, nw, nh);
    const px = wctx.getImageData(0, 0, SIZE, SIZE).data;
    const n = SIZE * SIZE, out = new Float32Array(3 * n);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      out[i] = px[p] / 255; out[n + i] = px[p + 1] / 255; out[2 * n + i] = px[p + 2] / 255;
    }
    return { tensor: new ort.Tensor('float32', out, [1, 3, SIZE, SIZE]), w, h, r, dx, dy };
  }

  // ---------- postprocess ----------
  function iou(a, b) {
    const iw = Math.min(a.box[2], b.box[2]) - Math.max(a.box[0], b.box[0]);
    const ih = Math.min(a.box[3], b.box[3]) - Math.max(a.box[1], b.box[1]);
    if (iw <= 0 || ih <= 0) return 0;
    const inter = iw * ih;
    const ua = (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]) + (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]) - inter;
    return ua > 0 ? inter / ua : 0;
  }

  function postprocess(out, labels, conf, pre, drop) {
    const [, C, N] = out.dims;
    const d = out.data, nc = labels.length;
    const cand = [];
    const skip = drop && drop.size ? labels.map((l) => drop.has(l)) : null;
    for (let i = 0; i < N; i++) {
      let best = -1, bs = conf;
      for (let k = 0; k < nc; k++) {
        if (skip && skip[k]) continue;
        const s = d[(4 + k) * N + i];
        if (s > bs) { bs = s; best = k; }
      }
      if (best < 0) continue;
      const cx = d[i], cy = d[N + i], bw = d[2 * N + i], bh = d[3 * N + i];
      const x1 = Math.max(0, Math.min(pre.w, (cx - bw / 2 - pre.dx) / pre.r));
      const y1 = Math.max(0, Math.min(pre.h, (cy - bh / 2 - pre.dy) / pre.r));
      const x2 = Math.max(0, Math.min(pre.w, (cx + bw / 2 - pre.dx) / pre.r));
      const y2 = Math.max(0, Math.min(pre.h, (cy + bh / 2 - pre.dy) / pre.r));
      cand.push({ label: labels[best], confidence: bs, box: [x1, y1, x2, y2] });
    }
    cand.sort((a, b) => b.confidence - a.confidence);
    const keep = [];
    for (const c of cand) {
      if (keep.length >= MAX_DET) break;
      if (keep.every((k) => iou(k, c) < IOU)) keep.push(c); // class-agnostic NMS
    }
    return keep;
  }

  // ---------- scene description (port of scene.py) ----------
  const NUMBERS = { 1: 'a', 2: 'two', 3: 'three', 4: 'four', 5: 'five' };
  const IRREGULAR = { person: 'people', knife: 'knives', mouse: 'mice', bus: 'buses', tv: 'TVs', sheep: 'sheep', fish: 'fish' };
  const PAIR = new Set(['scissors', 'skis', 'pants', 'glasses']);

  function pluralWord(w) {
    if (IRREGULAR[w]) return IRREGULAR[w];
    if (PAIR.has(w)) return w;
    if (/(s|x|z|ch|sh)$/.test(w)) return w + 'es';
    if (w.length > 1 && w.endsWith('y') && !'aeiou'.includes(w[w.length - 2])) return w.slice(0, -1) + 'ies';
    return w + 's';
  }
  function plural(label, n) {
    if (n === 1) return PAIR.has(label) ? `a pair of ${label}` : label;
    const i = label.lastIndexOf(' ');
    return i < 0 ? pluralWord(label) : label.slice(0, i + 1) + pluralWord(label.slice(i + 1));
  }
  function phrase(label, n, pos, prox) {
    const prefix = prox === 'close' ? 'very close: ' : '';
    if (n === 1 && PAIR.has(label)) return `${prefix}${plural(label, 1)} ${pos}`;
    let count = NUMBERS[n] || 'several';
    if (count === 'a' && 'aeiou'.includes(label[0])) count = 'an';
    return `${prefix}${count} ${plural(label, n)} ${pos}`;
  }
  function describe(dets, w, h) {
    if (!dets.length) return 'Nothing detected';
    const groups = new Map();
    for (const d of dets) {
      const cx = (d.box[0] + d.box[2]) / 2 / w;
      const pos = cx < 1 / 3 ? 'on the left' : cx > 2 / 3 ? 'on the right' : 'in front';
      const ratio = ((d.box[2] - d.box[0]) * (d.box[3] - d.box[1])) / (w * h);
      const prox = ratio >= CLOSE ? 'close' : ratio >= NEAR ? 'near' : 'far';
      const key = d.label + '\u0000' + pos;
      if (!groups.has(key)) groups.set(key, { label: d.label, pos, proxes: [] });
      groups.get(key).proxes.push(prox);
    }
    const parts = [];
    for (const g of groups.values()) {
      const closest = g.proxes.includes('close') ? 'close' : g.proxes.includes('near') ? 'near' : 'far';
      parts.push([closest !== 'close', -g.proxes.length, phrase(g.label, g.proxes.length, g.pos, closest)]);
    }
    parts.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]) || (a[2] < b[2] ? -1 : a[2] > b[2] ? 1 : 0));
    return parts.slice(0, 4).map((p) => p[2]).join(', ');
  }

  function summarize(dets) {
    const g = new Map();
    for (const d of dets) { if (!g.has(d.label)) g.set(d.label, []); g.get(d.label).push(d.confidence); }
    return [...g].map(([label, c]) => ({ label, count: c.length, confidence: Math.round(Math.max(...c) * 1000) / 1000 }))
      .sort((a, b) => (b.count * b.confidence - a.count * a.confidence) || (a.label < b.label ? -1 : 1));
  }

  // ---------- public API ----------
  let queue = Promise.resolve(); // one inference at a time
  function analyze(src, kind = 'live', onStatus) {
    const job = queue.then(async () => {
      const { session, labels, conf, person, drop, merge } = await load(kind, onStatus);
      const pre = preprocess(src);
      const res = await session.run({ [session.inputNames[0]]: pre.tensor });
      let dets = postprocess(res[session.outputNames[0]], labels, conf, pre, drop);
      if (kind === 'lens') dets = dets.map((d) => ({ ...d, label: person.has(d.label) ? 'person' : merge[d.label] || ALIASES[d.label] || d.label }));
      dets = dets.map((d) => ({ label: d.label, confidence: Math.round(d.confidence * 1000) / 1000, box: d.box.map(Math.round) }));
      return { output: describe(dets, pre.w, pre.h), objects: summarize(dets), width: pre.w, height: pre.h, detections: dets };
    });
    queue = job.catch(() => {});
    return job;
  }

  window.Vision = { analyze, preload: (kind, onStatus) => load(kind, onStatus), backend: hasGPU ? 'webgpu' : 'wasm' };
})();
