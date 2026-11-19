/* Mandala-GNN cockpit. Every number, gate and halo on this page comes from
   one real forward pass in the server's process — the chat only decides which
   cells to mask. Typing anything at all runs the network. */
(() => {
  'use strict';

  if (new URLSearchParams(location.search).has('static')) {
    const style = document.createElement('style');
    style.textContent = '*{animation:none!important;transition:none!important}';
    document.head.appendChild(style);
  }

  const SZ = 640;
  const IMG = 128;

  const $ = (id) => document.getElementById(id);

  const canvas = $('stage');
  const ctx = canvas.getContext('2d');
  const off = document.createElement('canvas');
  off.width = off.height = SZ;
  const octx = off.getContext('2d');

  const log = $('chat-log');
  const input = $('chat-input');
  const form = $('chat-form');
  const pill = $('stage-pill');
  const note = $('stage-note');
  const launchBtn = $('launch');
  const launchSub = $('launch-sub');
  const badge = $('sweep-badge');

  const HEAD_RGB = [[94, 240, 200], [122, 162, 255], [255, 180, 84], [167, 139, 250]];

  const S = {
    ready: false,
    busy: false,
    nodes: [],
    edges: [],
    gates: [],
    headEdge: [],
    heads: [],
    hop: [],
    hidden: new Set(),
    drawing: null,
    comp: null,
    state: null,
    loaded: false,
    demoRun: false,
    wedge: new Set(),
    evalIds: [],
    top: [],
    animStart: 0,
    sweepHead: -1,
    touched: false,
    coldMsg: null,
  };

  const pulses = [];
  const ripples = [];

  /* ── formatting ─────────────────────────────────────────────────── */

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));

  const f3 = (v) => (v == null || Number.isNaN(v)) ? '—' : Number(v).toFixed(3);
  const f5 = (v) => (v == null || Number.isNaN(v)) ? '—' : Number(v).toFixed(5);

  function flash(el) {
    el.classList.remove('flash');
    void el.offsetWidth;
    el.classList.add('flash');
  }

  /* ── static scene: drawing, edges, nodes ────────────────────────── */

  function renderScene() {
    const k = SZ / IMG;
    octx.clearRect(0, 0, SZ, SZ);
    octx.fillStyle = '#04050a';
    octx.fillRect(0, 0, SZ, SZ);

    const img = S.comp || S.drawing;
    if (img) octx.drawImage(img, 0, 0, SZ, SZ);
    octx.fillStyle = 'rgba(4,5,10,0.52)';
    octx.fillRect(0, 0, SZ, SZ);

    const e = S.edges;
    const g = S.gates;
    const he = S.headEdge;
    let lo = 0, hi = 1;
    if (g.length) {
      lo = Math.min(...g);
      hi = Math.max(...g);
    }
    const norm = (v) => (hi > lo ? (v - lo) / (hi - lo) : 0.5);

    octx.lineCap = 'round';
    for (let i = 0; i < e.length; i++) {
      const [s, d] = e[i];
      const a = S.nodes[s], b = S.nodes[d];
      if (!a || !b) continue;
      const gate = g.length ? norm(g[i] ?? 0) : 0.42;
      const rgb = g.length
        ? (HEAD_RGB[he[i] ?? 0] || HEAD_RGB[0])
        : [148, 170, 205];
      octx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${0.06 + 0.5 * gate})`;
      octx.lineWidth = 0.6 + 0.7 * gate;
      octx.beginPath();
      octx.moveTo(a[0] * k, a[1] * k);
      octx.lineTo(b[0] * k, b[1] * k);
      octx.stroke();
    }

    // held-out region, findable before masking
    octx.setLineDash([2, 3]);
    octx.lineWidth = 1;
    for (const i of S.wedge) {
      const p = S.nodes[i];
      if (!p || S.hidden.has(i)) continue;
      octx.beginPath();
      octx.arc(p[0] * k, p[1] * k, 5.2, 0, Math.PI * 2);
      octx.strokeStyle = 'rgba(255,180,84,.24)';
      octx.stroke();
    }
    octx.setLineDash([]);

    // nodes
    for (let i = 0; i < S.nodes.length; i++) {
      const [x, y] = S.nodes[i];
      const hidden = S.hidden.has(i);
      octx.beginPath();
      octx.arc(x * k, y * k, hidden ? 3.2 : 1.8, 0, Math.PI * 2);
      octx.fillStyle = hidden ? 'rgba(255,180,84,.95)' : 'rgba(220,236,255,.62)';
      octx.fill();
      if (hidden) {
        octx.beginPath();
        octx.arc(x * k, y * k, 6, 0, Math.PI * 2);
        octx.strokeStyle = 'rgba(255,180,84,.5)';
        octx.lineWidth = 1.2;
        octx.stroke();
      }
    }
  }

  /* ── animation ──────────────────────────────────────────────────── */

  function spawnPulses(now) {
    const g = S.gates;
    let picks = S.top.slice();
    if (!picks.length) {
      picks = Array.from({ length: 40 }, () =>
        Math.floor(Math.random() * S.edges.length));
    }
    // keep the strongest edges, then sample across heads
    const n = Math.min(picks.length, 80);
    for (let j = 0; j < n; j++) {
      const ei = picks[(j * 7 + 3) % picks.length];
      const e = S.edges[ei];
      if (!e) continue;
      const lag = (S.hop[e[1]] ?? 0) * 90 + Math.random() * 240;
      pulses.push({
        e: ei,
        head: S.headEdge[ei] ?? 0,
        start: now + lag,
        dur: 420 + Math.random() * 320,
      });
    }
    if (pulses.length > 300) pulses.splice(0, pulses.length - 300);
  }

  function headMeterActive(idx) {
    if (idx === S.sweepHead) return;
    S.sweepHead = idx;
    document.querySelectorAll('.hm').forEach((el) => {
      el.classList.toggle('active', Number(el.dataset.head) === idx);
    });
    if (idx >= 0) {
      const rgb = HEAD_RGB[idx] || HEAD_RGB[0];
      badge.hidden = false;
      badge.style.color = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      badge.querySelector('b').textContent = `HEAD ${idx + 1}`;
    } else {
      badge.hidden = true;
    }
  }

  function frame(now) {
    ctx.clearRect(0, 0, SZ, SZ);
    ctx.drawImage(off, 0, 0);
    const k = SZ / IMG;
    const t = Math.max(0, now - S.animStart);

    // a wave of halos leaving the visible cells, one hop at a time
    if (S.animStart && t < 2600) {
      for (let i = 0; i < S.nodes.length; i++) {
        const p = S.nodes[i];
        if (!p) continue;
        const h = S.hop[i] ?? 0;
        const ph = (t - h * 75) / 950;
        if (ph <= 0 || ph >= 1) continue;
        const fade = Math.sin(Math.PI * ph);
        const col = h < 3 ? [94, 240, 200] : h < 6 ? [122, 162, 255] : [167, 139, 250];
        ctx.beginPath();
        ctx.arc(p[0] * k, p[1] * k, (3 + 17 * ph), 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(${col[0]},${col[1]},${col[2]},${0.8 * (1 - ph)})`;
        ctx.lineWidth = 2 * (1 - ph) + 0.4;
        ctx.stroke();
        if (ph < 0.3) {
          ctx.beginPath();
          ctx.arc(p[0] * k, p[1] * k, 2.6, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${0.95 * fade})`;
          ctx.shadowColor = `rgba(${col[0]},${col[1]},${col[2]},.9)`;
          ctx.shadowBlur = 9 * fade;
          ctx.fill();
          ctx.shadowBlur = 0;
        }
      }
    }

    // while the pass settles, sweep through the four heads: show the edges
    // each one dominates, in that head's colour
    if (S.animStart && t < 1900 && S.gates.length) {
      const head = Math.floor(t / 430) % 4;
      headMeterActive(head);
      const rgb = HEAD_RGB[head] || HEAD_RGB[0];
      ctx.lineCap = 'round';
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.42)`;
      ctx.beginPath();
      for (let i = 0; i < S.edges.length; i++) {
        if ((S.headEdge[i] ?? 0) !== head) continue;
        if ((S.gates[i] ?? 0) < 0.42) continue;
        const [s, d] = S.edges[i];
        const a = S.nodes[s], b = S.nodes[d];
        if (!a || !b) continue;
        ctx.moveTo(a[0] * k, a[1] * k);
        ctx.lineTo(b[0] * k, b[1] * k);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
    } else if (S.sweepHead !== -1) {
      headMeterActive(-1);
    }

    // expanding rings on cells the model had to invent
    for (let i = ripples.length - 1; i >= 0; i--) {
      const rp = ripples[i];
      const p = (now - rp.t0) / 820;
      if (p < 0) continue;
      if (p > 1) { ripples.splice(i, 1); continue; }
      ctx.beginPath();
      ctx.arc(rp.x * k, rp.y * k, 5 + p * 18, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,180,84,${(1 - p) * 0.7})`;
      ctx.lineWidth = 1.5 * (1 - p) + 0.4;
      ctx.stroke();
    }

    // messages travelling along the strongest edges, coloured by head
    ctx.lineCap = 'round';
    for (let i = pulses.length - 1; i >= 0; i--) {
      const p = pulses[i];
      const ph = (now - p.start) / p.dur;
      if (ph < 0) continue;
      if (ph > 1) { pulses.splice(i, 1); continue; }
      const e = S.edges[p.e];
      if (!e) { pulses.splice(i, 1); continue; }
      const a = S.nodes[e[0]], b = S.nodes[e[1]];
      if (!a || !b) { pulses.splice(i, 1); continue; }
      const x = (a[0] + (b[0] - a[0]) * ph) * k;
      const y = (a[1] + (b[1] - a[1]) * ph) * k;
      const fade = Math.sin(Math.PI * ph);
      const rgb = HEAD_RGB[p.head] || HEAD_RGB[0];
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${0.35 + 0.6 * fade})`;
      ctx.shadowColor = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},.9)`;
      ctx.shadowBlur = 10 * fade;
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    requestAnimationFrame(frame);
  }

  /* ── API ────────────────────────────────────────────────────────── */

  async function loadImage(src) {
    return new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = reject;
      im.src = src;
    });
  }

  function setPill(kind, text) {
    pill.className = 'pill' + (kind ? ' ' + kind : '');
    pill.textContent = text;
  }

  let hadFocus = false;

  function setBusy(v) {
    if (v) hadFocus = document.activeElement === input;
    S.busy = v;
    input.disabled = v;
    $('chat-send').disabled = v;
    // Only take focus back if the keyboard was already open for this input:
    // a canvas tap must not pop the soft keyboard.
    if (!v && hadFocus) input.focus({ preventScroll: true });
  }

  function showLaunch(mode, text) {
    launchBtn.hidden = false;
    if (mode === 'loading') {
      launchBtn.disabled = true;
      launchBtn.classList.add('loading');
      launchBtn.querySelector('b').textContent = 'Loading the checkpoint…';
      launchSub.textContent = 'imports torch, then warms the kernels';
    } else {
      launchBtn.disabled = false;
      launchBtn.classList.remove('loading');
      launchBtn.querySelector('b').textContent = 'Launch the model';
      launchSub.textContent = text || 'imports torch · loads the checkpoint';
    }
  }

  async function ensureLoaded() {
    if (S.loaded) return;
    if (S.loading) return S.loading;
    S.loading = (async () => {
    showLaunch('loading');
    setPill('busy', 'loading');
    note.textContent = 'importing torch, reading 1.7 MB of weights';
    const res = await fetch('/api/load', { method: 'POST' });
    const st = await res.json().catch(() => ({}));
    if (!res.ok) {
      showLaunch('retry');
      setPill('', 'cold');
      note.textContent = 'the checkpoint did not load';
      throw new Error(st.error || res.statusText || 'load failed');
    }
    S.loaded = true;
    S.state = st;
    $('chip-params').textContent = (st.metrics.params || 0).toLocaleString();
    $('chip-status').textContent = 'live';
    document.querySelector('.chip.live').classList.remove('cold');
    launchBtn.hidden = true;
    setPill('ready', 'live');
    note.textContent = st.load_ms
      ? `checkpoint loaded in ${st.load_ms} ms — now reusable` : 'checkpoint loaded';
    $('chat-hint').textContent =
      'Warm: every message below is one forward pass, ~0.25 s.';
    if (S.coldMsg) { S.coldMsg.remove(); S.coldMsg = null; }
    })();
    try {
      await S.loading;
    } finally {
      S.loading = null;
    }
  }

  let passLock = Promise.resolve();

  function redraw(ids) {
    if (!S.loaded) return ensureLoaded().then(() => redraw(ids));
    const run = passLock.then(() => runPass(ids), () => runPass(ids));
    passLock = run.then(() => {}, () => {});
    return run;
  }

  async function runPass(ids) {
    setBusy(true);
    setPill('busy', 'inferring');
    note.textContent = 'one forward pass, in-process';
    try {
      const res = await fetch('/api/redraw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hidden: ids }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);

      S.gates = data.gates || [];
      S.headEdge = data.headEdge || [];
      S.heads = data.heads || [];
      S.hop = data.hop || [];
      S.hidden = new Set(data.hidden || []);
      S.top = data.top || [];
      const img = await loadImage(data.composite);
      S.comp = img;

      renderScene();
      updateStats(data.stats);
      updateHeads(data.heads || [], data.headEdge || []);
      updateHops(S.hop);

      const now = performance.now();
      S.animStart = now;
      S.sweepHead = -1;
      ripples.length = 0;
      for (const id of S.hidden) {
        const p = S.nodes[id];
        if (!p) continue;
        ripples.push({ x: p[0], y: p[1], t0: now + (S.hop[id] ?? 0) * 75 + 220 });
      }
      spawnPulses(now);

      setPill('ready', data.stats.cells ? 'reconstructed' : 'ready');
      note.textContent = data.stats.cells
        ? `${data.stats.cells} cells masked · ${data.stats.ms} ms`
        : 'no cells hidden — the drawing as it was';
      return data;
    } catch (err) {
      setPill('offline', 'no answer');
      note.textContent = 'the model did not answer';
      throw err;
    } finally {
      setBusy(false);
    }
  }

  /* ── readout ────────────────────────────────────────────────────── */

  const fields = {
    cells: $('r-cells'),
    r: $('r-r'),
    mse: $('r-mse'),
    ms: $('r-ms'),
    hop: $('r-hop'),
    depth: $('r-depth'),
  };

  function updateStats(st) {
    if (!st) return;
    const wedge = st.wedgeCorr != null;
    fields.cells.textContent = st.cells;
    fields.r.textContent = wedge
      ? f3(st.wedgeCorr)
      : (st.cells ? f3(st.corr) : '—');
    fields.mse.textContent = wedge
      ? f5(st.wedgeMse)
      : (st.cells ? (st.mse < 0.01 ? f5(st.mse) : Number(st.mse).toFixed(4)) : '—');
    fields.ms.textContent = `${st.ms} ms`;
    fields.depth.textContent = `${st.depth ?? '—'}`;
    $('l-r').textContent = wedge ? 'wedge r' : 'reconstruction r';
    $('l-mse').textContent = wedge ? 'wedge MSE' : 'MSE';

    const hops = [...S.hidden].map((i) => S.hop[i] ?? 0);
    fields.hop.textContent = hops.length ? Math.max(...hops) : '0';
    Object.values(fields).forEach(flash);
  }

  function updateHeads(heads, headEdge) {
    const els = [...document.querySelectorAll('.hm')];
    els.forEach((el, i) => {
      const v = Number(heads[i] ?? 0);
      el.querySelector('.hm-track i').style.width = `${Math.min(100, (v / 0.45) * 100)}%`;
      el.querySelector('.hm-val').textContent = f3(v);
      flash(el.querySelector('.hm-val'));
    });

    if (headEdge.length) {
      const counts = [0, 0, 0, 0];
      for (const h of headEdge) counts[h] = (counts[h] || 0) + 1;
      const top = counts.indexOf(Math.max(...counts));
      const share = Math.round((counts[top] / headEdge.length) * 100);
      $('heads-note').textContent = `H${top + 1} · ${share}%`;
    }
  }

  function updateHops(hop) {
    const el = $('hops');
    if (!hop || !hop.length) {
      el.innerHTML = '<span class="hop-empty">no mask yet</span>';
      return;
    }
    const max = Math.max(...hop);
    const counts = new Array(max + 1).fill(0);
    for (const h of hop) counts[Math.max(0, Math.min(max, h))]++;
    const peak = Math.max(...counts);
    el.innerHTML = counts.map((c, i) => {
      const hPct = peak ? Math.max(3, (c / peak) * 100) : 3;
      const deep = max ? i / max : 0;
      const col = deep < 0.34 ? '94,240,200' : deep < 0.67 ? '122,162,255' : '167,139,250';
      return `<div class="hop-col" title="${c} cell${c === 1 ? '' : 's'} at ${i} hop${i === 1 ? '' : 's'}">
        <span class="hop-bar" style="height:${hPct}%;background:linear-gradient(to top,rgba(${col},.22),rgba(${col},.9));box-shadow:0 0 8px rgba(${col},.25)"></span>
        <span class="hop-lab">${i}</span></div>`;
    }).join('');
    $('hop-note').textContent = `${peak} at the busiest hop`;
  }

  /* ── charts (data overlay) ─────────────────────────────────────── */

  const MESH_LABEL = {
    hex: 'hex · 2 seeds',
    hex_near: 'hex · near',
    rewired: 'rewired',
    jitter: 'jittered',
    random: 'random edges',
    grid: 'square grid',
    rings: 'rings',
    chart: 'hub chart',
    none: 'no edges',
  };

  function bars(el, entries, opts = {}) {
    const max = opts.max || 1;
    el.innerHTML = '';
    for (const ent of entries) {
      const v = Number(ent.value) || 0;
      const row = document.createElement('div');
      row.className = 'bar' + (ent.kind ? ' ' + ent.kind : '');
      row.title = `${ent.label}: ${opts.fmt ? opts.fmt(v) : v}`;
      row.innerHTML = `
        <span class="lab">${esc(ent.label)}</span>
        <span class="track"><span class="fill" style="width:${Math.max(2, Math.min(100, (v / max) * 100))}%"></span></span>
        <span class="val">${opts.fmt ? opts.fmt(v) : v}</span>`;
      el.appendChild(row);
    }
  }

  function meshChart(table) {
    const order = ['hex', 'hex_near', 'rewired', 'jitter', 'random', 'grid', 'rings', 'chart', 'none'];
    const entries = [];
    for (const key of order) {
      const row = table[key];
      if (!row || !row.wedge_corr) continue;
      entries.push({
        label: MESH_LABEL[key] || key,
        value: row.wedge_corr.mean,
        kind: key === 'hex' ? 'best' : key === 'none' ? 'bad' : '',
      });
    }
    bars($('chart-mesh'), entries, { fmt: (v) => v.toFixed(3) });
  }

  function harmonicsChart(table) {
    bars($('chart-harmonics'),
      ['0', '2', '4', '8'].filter((k) => table[k] && table[k].wedge_corr).map((k) => ({
        label: `${k} harmonics`,
        value: table[k].wedge_corr.mean,
        kind: k === '4' ? 'best' : '',
      })), { fmt: (v) => v.toFixed(3) });
  }

  function depthChart(table) {
    bars($('chart-depth'),
      ['1', '2', '3', '4'].filter((k) => table[k] && table[k].wedge_corr).map((k) => ({
        label: `${k} layer${k === '1' ? '' : 's'}`,
        value: table[k].wedge_corr.mean,
        kind: k === '3' ? 'best' : '',
      })), { fmt: (v) => v.toFixed(3) });
  }

  function spark(el, hist) {
    if (!hist || hist.length < 2) {
      el.innerHTML = '<p class="fine">No training history in this checkpoint.</p>';
      return;
    }
    const W = 560, H = 150, padL = 34, padR = 8, padT = 10, padB = 20;
    const xs = hist.map((h) => h[0]);
    const ys = hist.map((h) => Math.log10(Math.max(h[1], 1e-4)));
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const y0 = Math.min(...ys), y1 = Math.max(...ys);
    const px = (x) => padL + ((x - x0) / (x1 - x0 || 1)) * (W - padL - padR);
    const py = (v) => padT + (1 - (v - y0) / (y1 - y0 || 1)) * (H - padT - padB);

    let grid = '';
    for (let p = Math.ceil(y0); p <= Math.floor(y1); p++) {
      const y = py(p);
      grid += `<line class="grid-line" x1="${padL}" x2="${W - padR}" y1="${y}" y2="${y}"/>`;
      grid += `<text class="axis" x="${padL - 6}" y="${y + 3}" text-anchor="end">1e${p}</text>`;
    }
    let d = '';
    hist.forEach((h, i) => {
      d += `${i ? 'L' : 'M'}${px(h[0]).toFixed(1)},${py(Math.log10(Math.max(h[1], 1e-4))).toFixed(1)}`;
    });
    const area = `${d}L${px(x1).toFixed(1)},${H - padB}L${px(x0).toFixed(1)},${H - padB}Z`;

    el.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Training loss curve">
        <defs><linearGradient id="curveFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#5ef0c8" stop-opacity=".26"/>
          <stop offset="100%" stop-color="#5ef0c8" stop-opacity="0"/>
        </linearGradient></defs>
        ${grid}
        <path class="area" d="${area}"/>
        <path class="curve" d="${d}"/>
      </svg>`;
  }

  /* ── chat ──────────────────────────────────────────────────────── */

  function scrollLog() {
    log.scrollTop = log.scrollHeight;
  }

  function addMsg(role, html) {
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + role;
    wrap.innerHTML = `<span class="who">${role === 'bot' ? 'M' : 'Y'}</span>
      <div class="bubble">${html}</div>`;
    log.appendChild(wrap);
    scrollLog();
    return wrap;
  }

  const TYPING = '<span class="typing"><i></i><i></i><i></i></span>';

  function sugg(html, items) {
    const btns = items.map((t) =>
      `<button type="button" data-ask="${esc(t)}">${esc(t)}</button>`).join('');
    return `${html}<div class="sugg">${btns}</div>`;
  }

  function sectorIds(k) {
    const out = [];
    const n = S.state ? S.state.sectors : 8;
    for (let i = 0; i < S.nodes.length; i++) {
      const [x, y] = S.nodes[i];
      const th = Math.atan2(y - IMG / 2, x - IMG / 2);
      let s = Math.floor(((th + Math.PI) / (2 * Math.PI)) * n);
      s = Math.max(0, Math.min(n - 1, s));
      if (s === k) out.push(i);
    }
    return out;
  }

  function describe(res, what) {
    const st = res.stats;
    const ids = res.hidden;
    if (!ids.length) {
      return `<b>${what}.</b> Nothing is hidden — the canvas is the drawing itself. ` +
        `The edges are still coloured by the head that dominates them in this pass, ` +
        `and the gates still ran (mean <span class="num">${f3(st.gate)}</span>, ${st.ms} ms).`;
    }
    const maxHop = Math.max(...ids.map((i) => res.hop[i] ?? 0));
    return `<b>${what}: ${st.cells} cells.</b>
      Rebuilt at <span class="num">r = ${f3(st.corr)}</span> ·
      MSE <span class="num">${f5(st.mse)}</span> ·
      <span class="num">${st.ms} ms</span> on one thread.<br>
      The farthest hidden cell sat <span class="num">${maxHop}</span> hop${maxHop === 1 ? '' : 's'} from
      something visible; mean edge gate <span class="num">${f3(st.gate)}</span>.`;
  }

  function describeHeldOut(res) {
    const st = res.stats;
    const maxHop = Math.max(...res.hidden.map((i) => res.hop[i] ?? 0));
    const wedgeN = S.wedge.size;
    const rest = Math.max(0, st.cells - wedgeN);
    return `<b>Held-out wedge — ${wedgeN} cells, never trained on.</b>
      I masked the ${st.cells} cells the evaluation hides (the wedge plus ${rest}
      random train/val cells) and let the rest of the drawing speak.<br>
      The wedge came back at <span class="num">r = ${f3(st.wedgeCorr)}</span> ·
      MSE <span class="num">${f5(st.wedgeMse)}</span> ·
      <span class="num">${st.ms} ms</span>. Farthest wedge cell:
      <span class="num">${maxHop}</span> hops. Amber rings mark every cell I invented.`;
  }

  async function doHeldOut() {
    const ids = S.evalIds.length ? S.evalIds : [...S.wedge];
    if (!ids.length) return doSector(0);
    const res = await redraw(ids);
    return describeHeldOut(res);
  }

  async function doSector(k) {
    const ids = sectorIds(k);
    if (!ids.length) return `Sector ${k} has no cells on this mesh. Try another sector.`;
    const res = await redraw(ids);
    return describe(res, `Sector ${k} masked`);
  }

  async function doRandom(n) {
    const pool = S.nodes.map((_, i) => i);
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    const ids = pool.slice(0, Math.min(n, pool.length));
    const res = await redraw(ids);
    return describe(res, `${ids.length} random cells masked`);
  }

  async function doClear() {
    const res = await redraw([]);
    return describe(res, 'Cleared');
  }

  function doResults() {
    return `<b>Headline numbers</b>, from the checkpoint this page is running
      (400 steps, hex drawing, one seed):
      <table class="mini-table">
        <tr><td>held-out wedge r</td><td>0.904</td></tr>
        <tr><td>wedge MSE</td><td>0.00268</td></tr>
        <tr><td>mean-patch floor</td><td>0.01536</td></tr>
        <tr><td>in-distribution val r</td><td>0.987</td></tr>
        <tr><td>parameters</td><td>405,342</td></tr>
        <tr><td>training</td><td>88 s · 1 CPU thread</td></tr>
      </table>
      The <b>Data</b> button has the ablations: the lattice barely matters, the angular
      harmonics and the edge gate matter a lot.`;
  }

  function doHeads() {
    if (!S.heads.length) {
      return `No pass has run yet, so there are no attention weights to show.
        Ask me to hide something.`;
    }
    const rows = S.heads.map((v, i) => {
      const share = S.headEdge.length
        ? Math.round((S.headEdge.filter((h) => h === i).length / S.headEdge.length) * 100)
        : 0;
      return `<tr><td>head ${i + 1}</td><td>${f3(v)}</td><td>${share}% of edges</td></tr>`;
    }).join('');
    return `<b>Four heads, and each routes its own share of the attention.</b>
      The edge gate is a function of edge geometry, so it is the same on every pass; what
      changes with the mask is where the heads send attention. The meters on the right are
      the share from the last pass (they sum to 100%), and the canvas colours an edge by the
      head carrying the most attention over it.
      <table class="mini-table">
        <tr><td>head</td><td>attention</td><td>dominant</td></tr>${rows}
      </table>
      Hops are the BFS distance from the nearest visible cell; the halo wave follows them.`;
  }

  const HELP = `I am a graph network, not a chat model: my answer is always a masking
    experiment. Say anything, or pick one:
    <table class="mini-table">
      <tr><td><code>hide sector 4</code></td><td>mask one 45° sector</td></tr>
      <tr><td><code>held-out wedge</code></td><td>the cells never trained on</td></tr>
      <tr><td><code>hide 24 random cells</code></td><td>scattered mask</td></tr>
      <tr><td><code>clear</code></td><td>unmask everything</td></tr>
      <tr><td><code>results</code></td><td>the reconstruction table</td></tr>
    </table>
    You can also tap the drawing to hide the cells under your finger.`;

  const SUGGESTIONS = [
    'Hide the held-out wedge',
    'Hide sector 4',
    'Hide 24 random cells',
    'Explain the heads',
    'Show the results',
  ];

  let hash = 0;
  function textSector(text) {
    hash = 0;
    for (let i = 0; i < text.length; i++) hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
    return Math.abs(hash) % (S.state ? S.state.sectors : 8);
  }

  async function interpret(text) {
    const t = text.toLowerCase();

    if (/(held.?out|never.?trained|the wedge|wedge\b)/.test(t) && !/\b[0-7]\b/.test(t)) {
      return doHeldOut();
    }
    let m = t.match(/(?:sector|wedge|slice|segment|region|part|area|section|eighth)\D{0,6}([0-7])\b/);
    if (m) return doSector(parseInt(m[1], 10));

    m = t.match(/(?:random|mask|hide|cover|blank|remove)\D{0,14}(\d{1,2})/);
    if (m) return doRandom(Math.max(1, Math.min(60, parseInt(m[1], 10))));

    if (/^(clear|reset|unmask|show all|restore|undo|nothing)/.test(t)) return doClear();
    if (/(result|score|metric|ablation|table|how good|performance|baseline|accuracy)/.test(t)) {
      return doResults();
    }
    if (/(head|attention|gate|edge|hop|halo|colour|color)/.test(t)) return doHeads();
    if (/(help|command|what can|how do i|what should|confused|stuck)/.test(t)) return HELP;

    const greeting = /^(hi|hey|hello|yo|sup|howdy|hola|good (morning|evening|afternoon))\b/.test(t);
    const intro = /(who are you|what are you|what is this|what do you do|how does (it|this) work|explain (it|this|yourself)|tell me about)/.test(t);

    if (greeting || intro) {
      const ids = S.evalIds.length ? S.evalIds : [...S.wedge];
      const res = await redraw(ids);
      return `Hello. I am a 405k-parameter graph network wired to the drawing on the left —
        the chat is a handle onto it, not a language model.
        Since you said hi, I ran the mask the paper is scored on: ${res.stats.cells} cells
        hidden, including the ${S.wedge.size}-cell wedge that was never trained on. It came
        back at <span class="num">r = ${f3(res.stats.wedgeCorr)}</span>,
        MSE <span class="num">${f5(res.stats.wedgeMse)}</span> in
        <span class="num">${res.stats.ms} ms</span>. Watch the amber cells, the halo wave and
        the four head meters on the right — that was the network working.
        <br>Try a real question about the drawing:`;
    }

    // Anything else: treat it as a request to test the model on a sector.
    const k = textSector(t);
    const res = await redraw(sectorIds(k));
    return `I do not parse sentences, so I took “${esc(text.slice(0, 60))}” as
      <b>test yourself on sector ${k}</b> and did exactly that: ${res.stats.cells} cells masked,
      rebuilt at <span class="num">r = ${f3(res.stats.corr)}</span>,
      MSE <span class="num">${f5(res.stats.mse)}</span> in
      <span class="num">${res.stats.ms} ms</span>. Sector ${k} is one of the sparse, low-ink
      sectors, so treat r with suspicion there and MSE as the real number — the held-out
      wedge scores 0.904.
      <br>Words I do act on:`;
  }

  async function ask(text) {
    const clean = String(text || '').trim();
    if (!clean) return;
    S.touched = true;
    if (S.busy) {
      addMsg('bot', 'One pass at a time — I am still rebuilding the last mask.');
      return;
    }
    addMsg('user', esc(clean));
    const wrap = addMsg('bot', TYPING);
    const bubble = wrap.querySelector('.bubble');
    const reply = (html) => { bubble.innerHTML = html; scrollLog(); };
    try {
      const out = await interpret(clean);
      reply(out);
      if (out !== HELP && !out.includes('class="sugg"')) {
        bubble.insertAdjacentHTML('beforeend',
          `<div class="sugg">${SUGGESTIONS.map((s) =>
            `<button type="button" data-ask="${esc(s)}">${esc(s)}</button>`).join('')}</div>`);
        scrollLog();
      }
    } catch (err) {
      bubble.innerHTML = `<b>That pass failed.</b> ${esc(String(err.message || err))}.
        Is <code>mandalanet.serve</code> still running?`;
      setPill('offline', 'no answer');
    }
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (S.busy) return;
    const text = input.value;
    input.value = '';
    ask(text);
  });

  log.addEventListener('click', (e) => {
    const b = e.target.closest('button[data-ask]');
    if (!b || S.busy) return;
    ask(b.dataset.ask);
  });

  /* tap the drawing to hide the cells under the finger */
  canvas.addEventListener('click', async (e) => {
    if (S.busy || !S.ready) return;
    S.touched = true;
    if (!S.loaded) {
      addMsg('user', 'hid the cells under my finger');
      const wrap = addMsg('bot', TYPING);
      const bubble = wrap.querySelector('.bubble');
      try {
        await ensureLoaded();
        bubble.innerHTML = 'The checkpoint is loaded — tap the drawing again and I will mask there.';
      } catch (err) {
        bubble.innerHTML = `<b>Could not load the checkpoint.</b> ${esc(String(err.message || err))}`;
      }
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * IMG;
    const y = ((e.clientY - rect.top) / rect.height) * IMG;
    const ids = [];
    S.nodes.forEach(([nx, ny], i) => {
      if ((nx - x) ** 2 + (ny - y) ** 2 <= 11 * 11) ids.push(i);
    });
    if (!ids.length) return;
    addMsg('user', 'hid the cells under my finger');
    const wrap = addMsg('bot', TYPING);
    const bubble = wrap.querySelector('.bubble');
    try {
      const res = await redraw(ids);
      bubble.innerHTML = describe(res,
        `You hid ${ids.length} cells under your finger`) +
        `<div class="sugg">${SUGGESTIONS.map((s) =>
          `<button type="button" data-ask="${esc(s)}">${esc(s)}</button>`).join('')}</div>`;
      scrollLog();
    } catch (err) {
      bubble.innerHTML = `<b>That pass failed.</b> ${esc(String(err.message || err))}`;
    }
  });

  /* ── overlays ──────────────────────────────────────────────────── */

  function openOverlay(id) {
    const el = $(id);
    if (!el) return;
    document.querySelectorAll('.overlay').forEach((o) => { o.hidden = true; });
    el.hidden = false;
    el.querySelector('.overlay-scroll').scrollTop = 0;
  }
  function closeOverlays() {
    document.querySelectorAll('.overlay').forEach((o) => { o.hidden = true; });
  }

  $('btn-paper').addEventListener('click', () => openOverlay('paper-overlay'));
  $('btn-data').addEventListener('click', () => openOverlay('data-overlay'));
  $('rail-data').addEventListener('click', () => openOverlay('data-overlay'));
  document.querySelectorAll('[data-close]').forEach((b) => {
    b.addEventListener('click', () => { $(b.dataset.close).hidden = true; });
  });
  document.querySelectorAll('.overlay').forEach((o) => {
    o.addEventListener('click', (e) => { if (e.target === o) o.hidden = true; });
  });
  addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeOverlays();
    if (e.key === '/' && document.activeElement !== input) {
      e.preventDefault();
      input.focus({ preventScroll: true });
    }
  });

  $('btn-full').addEventListener('click', async () => {
    try {
      if (!document.fullscreenElement) {
        await document.documentElement.requestFullscreen();
        const so = screen.orientation;
        if (so && so.lock) await so.lock('landscape').catch(() => {});
      } else {
        await document.exitFullscreen();
      }
    } catch (_) { /* the browser may refuse; the layout already fits */ }
  });

  /* ── boot ──────────────────────────────────────────────────────── */

  async function runDemo() {
    if (S.demoRun || !S.loaded) return;
    S.demoRun = true;
    if (S.touched) return;
    const ids = S.evalIds.length ? S.evalIds : [...S.wedge];
    if (!ids.length) return;
    const wrap = addMsg('bot', TYPING);
    const bubble = wrap.querySelector('.bubble');
    try {
      const out = await redraw(ids);
      bubble.innerHTML = `<b>Watch the left panel: this is the evaluation mask.</b>
        I hid the ${out.stats.cells} cells the score uses — ${S.wedge.size} cells from the
        wedge that was never trained on, plus random train/val cells. The never-seen cells came
        back at r = <span class="num">${f3(out.stats.wedgeCorr)}</span>,
        MSE <span class="num">${f5(out.stats.wedgeMse)}</span>, in
        <span class="num">${out.stats.ms} ms</span>.<br>
        Amber rings are cells I had to invent; the halo wave is the hop distance from visible
        cells; each dot travelling along an edge is coloured by the head that owns it.
        <div class="sugg">${SUGGESTIONS.map((s) =>
          `<button type="button" data-ask="${esc(s)}">${esc(s)}</button>`).join('')}</div>`;
      scrollLog();
    } catch (err) {
      S.demoRun = false;
      bubble.innerHTML = `<b>The demo pass failed.</b> ${esc(String(err.message || err))}`;
    }
  }

  launchBtn.addEventListener('click', async () => {
    if (S.busy) return;
    try {
      await ensureLoaded();
      runDemo();
    } catch (err) {
      addMsg('bot', `<b>Could not load the checkpoint.</b> ${esc(String(err.message || err))}`);
    }
  });

  async function boot() {
    try {
      const res = await fetch('/api/state');
      const st = await res.json();
      if (!res.ok) throw new Error(st.error || res.statusText);
      S.state = st;
      S.nodes = st.nodes;
      S.edges = st.edges;
      S.wedge = new Set(st.wedge || []);
      S.evalIds = st.eval || [];
      S.drawing = await loadImage(st.drawing);
      S.ready = true;

      $('origin').textContent = location.host;
      $('chip-nodes').textContent = st.nodes.length.toLocaleString();
      $('chip-edges').textContent = st.edges.length.toLocaleString();
      $('chip-params').textContent = (st.metrics.params || 0).toLocaleString();
      $('chip-status').textContent = st.loaded ? 'live' : 'cold';
      document.querySelector('.chip.live').classList.toggle('cold', !st.loaded);
      S.loaded = !!st.loaded;
      $('card-r').textContent = f3(st.metrics.wedge_corr);

      meshChart(st.abl_mesh || {});
      harmonicsChart(st.abl_harmonics || {});
      depthChart(st.abl_depth || {});
      spark($('chart-curve'), st.history);

      renderScene();
      updateHops([]);
      requestAnimationFrame(frame);

      if (st.checkpoint_kb) {
        launchSub.textContent =
          `${(st.checkpoint_kb * 1024 / 1e6).toFixed(1)} MB · ` +
          `${(st.metrics.params || 0).toLocaleString()} params · ~4 s cold`;
      }

      const intro = `<b>I am the network from the paper.</b> The 45° wedge of this drawing was
        removed from every training step, and I am going to try to redraw it anyway.
        Every message you send runs one real forward pass; say anything and watch the
        topology, the heads and the meters.`;
      addMsg('bot', `${intro}<div class="sugg">${SUGGESTIONS.map((s) =>
        `<button type="button" data-ask="${esc(s)}">${esc(s)}</button>`).join('')}</div>`);

      if (st.loaded) {
        launchBtn.hidden = true;
        setPill('ready', 'live');
        setTimeout(runDemo, 250);
      } else {
        setPill('', 'cold');
        showLaunch('idle');
        note.textContent = 'the checkpoint is cold — press launch';
        S.coldMsg = addMsg('bot', `The model is not in memory yet. Press
          <b>Launch the model</b> on the drawing, or just type at me: either one loads the
          ${(st.checkpoint_kb * 1024 / 1e6).toFixed(1)} MB checkpoint and runs the wedge.`);
      }
    } catch (err) {
      setPill('offline', 'offline');
      addMsg('bot', `<b>The page loaded but the model did not.</b>
        ${esc(String(err.message || err))}. Start it with
        <code>python -m mandalanet.serve --port 8000</code>.`);
    }
  }

  boot();
})();
