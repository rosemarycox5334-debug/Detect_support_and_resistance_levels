/* =========================================================================
   密位 · DENSITY — 背景动效（参考 AlphaMaster）
   1. 神经网络层 — 节点 + 连线 + 数据脉冲
   2. 数学符号层 — 支撑/概率相关符号缓慢漂移
   3. CSS 层负责极光 / 扫描线 / 暗角
   ========================================================================= */
(() => {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const neuralCanvas = document.getElementById("neuralCanvas");
  const glyphCanvas = document.getElementById("glyphCanvas");
  if (!neuralCanvas || !glyphCanvas) return;

  const nctx = neuralCanvas.getContext("2d");
  const gctx = glyphCanvas.getContext("2d");

  let W = 0, H = 0, DPR = Math.min(window.devicePixelRatio || 1, 2);
  const mouse = { x: 0.5, y: 0.35, tx: 0.5, ty: 0.35 };

  let nodes = [];
  let pulses = [];
  let glyphs = [];

  // 贴合本项目：概率 / ATR / 支撑阻力 / 金融数学
  const GLYPH_CHARS = [
    "∑", "∫", "∂", "∇", "σ", "μ", "ρ", "π", "λ", "θ", "Δ", "Ω",
    "√", "∞", "≈", "≠", "∝", "∈", "P", "ATR", "SR", "EMA",
    "Pₜ", "Pₕ", "×", "∮", "eˣ", "Φ", "Ψ", "τ",
  ];

  function rand(a, b) { return a + Math.random() * (b - a); }

  function computeCounts() {
    const area = W * H;
    return {
      nodeCount: Math.max(28, Math.min(90, Math.round(area / 22000))),
      glyphCount: Math.max(10, Math.min(28, Math.round(area / 70000))),
    };
  }

  function buildScene() {
    const { nodeCount, glyphCount } = computeCounts();
    nodes = new Array(nodeCount).fill(0).map(() => {
      const depth = rand(0.35, 1);
      return {
        x: rand(0, W), y: rand(0, H),
        vx: rand(-0.12, 0.12) * depth,
        vy: rand(-0.12, 0.12) * depth,
        depth, r: rand(1.1, 2.6) * depth,
        phase: rand(0, Math.PI * 2),
        pspeed: rand(0.6, 1.6),
      };
    });
    glyphs = new Array(glyphCount).fill(0).map(() => ({
      char: GLYPH_CHARS[(Math.random() * GLYPH_CHARS.length) | 0],
      x: rand(0, W), y: rand(0, H),
      size: rand(24, 78),
      vx: rand(-0.06, 0.06), vy: rand(-0.05, 0.05),
      rot: rand(0, Math.PI * 2),
      vrot: rand(-0.0025, 0.0025),
      alpha: rand(0.025, 0.08),
      pulse: rand(0, Math.PI * 2),
    }));
    pulses = [];
  }

  function resize() {
    W = window.innerWidth;
    H = window.innerHeight;
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    for (const c of [neuralCanvas, glyphCanvas]) {
      c.width = W * DPR;
      c.height = H * DPR;
      c.style.width = W + "px";
      c.style.height = H + "px";
    }
    nctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    gctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    buildScene();
  }

  const LINK_DIST = 168;
  const LINK_DIST_SQ = LINK_DIST * LINK_DIST;

  function spawnPulse(a, b) {
    pulses.push({
      ax: a.x, ay: a.y, bx: b.x, by: b.y,
      t: 0, speed: rand(0.006, 0.016),
      hue: Math.random() < 0.55 ? "cyan" : "mint",
    });
  }

  function maybeSpawnPulses() {
    if (reduceMotion || pulses.length > 46 || Math.random() > 0.28) return;
    const a = nodes[(Math.random() * nodes.length) | 0];
    let best = null, bestD = LINK_DIST_SQ;
    for (const b of nodes) {
      if (b === a) continue;
      const dx = a.x - b.x, dy = a.y - b.y;
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = b; }
    }
    if (best) spawnPulse(a, best);
  }

  let last = performance.now();
  let running = true;

  function frame(now) {
    if (!running) return;
    const dt = Math.min(48, now - last);
    last = now;
    mouse.x += (mouse.tx - mouse.x) * 0.05;
    mouse.y += (mouse.ty - mouse.y) * 0.05;
    const parX = mouse.x - 0.5;
    const parY = mouse.y - 0.5;
    drawGlyphs(dt, parX, parY);
    drawNeural(dt, now, parX, parY);
    requestAnimationFrame(frame);
  }

  function drawGlyphs(dt, parX, parY) {
    gctx.clearRect(0, 0, W, H);
    gctx.textAlign = "center";
    gctx.textBaseline = "middle";
    for (const g of glyphs) {
      if (!reduceMotion) {
        g.x += g.vx * (dt / 16);
        g.y += g.vy * (dt / 16);
        g.rot += g.vrot * (dt / 16);
        g.pulse += 0.01 * (dt / 16);
      }
      if (g.x < -100) g.x = W + 100;
      if (g.x > W + 100) g.x = -100;
      if (g.y < -100) g.y = H + 100;
      if (g.y > H + 100) g.y = -100;

      const px = g.x + parX * 34 * (g.size / 84);
      const py = g.y + parY * 34 * (g.size / 84);
      const a = g.alpha + Math.sin(g.pulse) * 0.02;

      gctx.save();
      gctx.translate(px, py);
      gctx.rotate(g.rot);
      gctx.font = `${g.size}px "JetBrains Mono", monospace`;
      gctx.fillStyle = `rgba(91, 203, 255, ${Math.max(0.014, a)})`;
      gctx.shadowColor = "rgba(56, 215, 255, 0.45)";
      gctx.shadowBlur = 12;
      gctx.fillText(g.char, 0, 0);
      gctx.restore();
    }
  }

  function drawNeural(dt, now, parX, parY) {
    nctx.clearRect(0, 0, W, H);

    for (const n of nodes) {
      if (!reduceMotion) {
        n.x += n.vx * (dt / 16);
        n.y += n.vy * (dt / 16);
      }
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;
      n.x = Math.max(0, Math.min(W, n.x));
      n.y = Math.max(0, Math.min(H, n.y));
    }

    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      const ax = a.x + parX * 18 * a.depth;
      const ay = a.y + parY * 18 * a.depth;
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const dsq = dx * dx + dy * dy;
        if (dsq > LINK_DIST_SQ) continue;
        const d = Math.sqrt(dsq);
        const strength = 1 - d / LINK_DIST;
        nctx.beginPath();
        nctx.moveTo(ax, ay);
        nctx.lineTo(b.x + parX * 18 * b.depth, b.y + parY * 18 * b.depth);
        nctx.strokeStyle = `rgba(56, 215, 255, ${strength * 0.13})`;
        nctx.lineWidth = strength * 1.1;
        nctx.stroke();
      }
    }

    for (const n of nodes) {
      const px = n.x + parX * 18 * n.depth;
      const py = n.y + parY * 18 * n.depth;
      const breath = reduceMotion ? 0.7
        : 0.55 + 0.45 * (0.5 + 0.5 * Math.sin(now * 0.001 * n.pspeed + n.phase));
      const r = n.r * (0.8 + breath * 0.5);
      const grd = nctx.createRadialGradient(px, py, 0, px, py, r * 5);
      grd.addColorStop(0, `rgba(180, 245, 255, ${0.48 * breath})`);
      grd.addColorStop(0.4, `rgba(56, 215, 255, ${0.15 * breath})`);
      grd.addColorStop(1, "rgba(56, 215, 255, 0)");
      nctx.fillStyle = grd;
      nctx.beginPath();
      nctx.arc(px, py, r * 5, 0, Math.PI * 2);
      nctx.fill();
      nctx.fillStyle = `rgba(220, 250, 255, ${0.75 * breath})`;
      nctx.beginPath();
      nctx.arc(px, py, r, 0, Math.PI * 2);
      nctx.fill();
    }

    maybeSpawnPulses();
    for (let i = pulses.length - 1; i >= 0; i--) {
      const p = pulses[i];
      p.t += p.speed * (dt / 16);
      if (p.t >= 1) { pulses.splice(i, 1); continue; }
      const x = p.ax + (p.bx - p.ax) * p.t + parX * 18;
      const y = p.ay + (p.by - p.ay) * p.t + parY * 18;
      const fade = Math.sin(p.t * Math.PI);
      const col = p.hue === "cyan" ? "56, 215, 255" : "57, 245, 166";
      const grd = nctx.createRadialGradient(x, y, 0, x, y, 7);
      grd.addColorStop(0, `rgba(${col}, ${0.9 * fade})`);
      grd.addColorStop(1, `rgba(${col}, 0)`);
      nctx.fillStyle = grd;
      nctx.beginPath();
      nctx.arc(x, y, 7, 0, Math.PI * 2);
      nctx.fill();
      nctx.fillStyle = `rgba(235, 255, 252, ${fade})`;
      nctx.beginPath();
      nctx.arc(x, y, 1.6, 0, Math.PI * 2);
      nctx.fill();
    }
  }

  window.addEventListener("pointermove", (e) => {
    mouse.tx = e.clientX / Math.max(W, 1);
    mouse.ty = e.clientY / Math.max(H, 1);
  }, { passive: true });

  window.addEventListener("resize", () => {
    clearTimeout(window.__bgResize);
    window.__bgResize = setTimeout(resize, 150);
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      running = false;
    } else if (!running) {
      running = true;
      last = performance.now();
      requestAnimationFrame(frame);
    }
  });

  resize();
  if (reduceMotion) {
    drawGlyphs(16, 0, 0);
    drawNeural(16, performance.now(), 0, 0);
  } else {
    requestAnimationFrame(frame);
  }
})();
