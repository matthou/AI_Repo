/**
 * Network Topology Visualizer — v2
 * Adds: clickable neurons, per-layer gradient heatmap,
 *       dropout indicators, cleaner gradient particle system.
 */

class NetworkVisualizer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.nn = null;
    this.animFrame = null;
    this.pulsePhase = 0;
    this.gradientParticles = [];
    this.hoveredNeuron = null;
    this.selectedNeuron = null;
    this.sampleActivations = null;
    this.onNeuronClick = null;   // callback(l, j)

    this._bindEvents();
  }

  setNetwork(nn) {
    this.nn = nn;
    this.gradientParticles = [];
    this.sampleActivations = null;
    this.selectedNeuron = null;
  }

  setSampleActivations(activations) {
    this.sampleActivations = activations;
  }

  // ─── Layout ──────────────────────────────────────────────────────────────

  _computeLayout() {
    const { width, height } = this.canvas;
    const nn = this.nn;
    const layerCount = nn.layerSizes.length;
    const padX = 80;
    const padY = 48;
    const usableW = width - padX * 2;
    const usableH = height - padY * 2;

    return nn.layerSizes.map((count, l) => {
      const x = layerCount === 1 ? padX : padX + (l / (layerCount - 1)) * usableW;
      const neurons = Array.from({ length: count }, (_, j) => ({
        x,
        y: padY + ((j + 0.5) / count) * usableH,
        l, j
      }));
      return { x, neurons, l };
    });
  }

  _neuronRadius(count) {
    const maxCount = Math.max(...this.nn.layerSizes);
    return Math.min(22, Math.max(7, 220 / Math.max(maxCount, 1)));
  }

  // ─── Particles ───────────────────────────────────────────────────────────

  spawnGradientParticles(layers) {
    if (!this.nn?.weightGradMagnitudes) return;

    for (let l = 0; l < layers.length - 1; l++) {
      const fromLayer = layers[l];
      const toLayer = layers[l + 1];
      const gradLayer = this.nn.weightGradMagnitudes[l];

      for (let j = 0; j < toLayer.neurons.length; j++) {
        for (let k = 0; k < fromLayer.neurons.length; k++) {
          const mag = gradLayer?.[j]?.[k] ?? 0;
          if (mag > 0.001 && Math.random() < Math.min(mag * 15, 0.35)) {
            const from = fromLayer.neurons[k];
            const to   = toLayer.neurons[j];
            this.gradientParticles.push({
              x: from.x, y: from.y, tx: to.x, ty: to.y,
              progress: 0,
              speed: 0.013 + Math.random() * 0.018,
              size: 1.5 + Math.min(mag * 40, 4),
              opacity: 0.9,
              hue: mag > 0.08 ? 20 : 245   // orange vs indigo
            });
          }
        }
      }
    }

    if (this.gradientParticles.length > 300) {
      this.gradientParticles = this.gradientParticles.slice(-300);
    }
  }

  _updateParticles() {
    for (const p of this.gradientParticles) {
      p.progress += p.speed;
      p.opacity = Math.max(0, 1 - p.progress);
    }
    this.gradientParticles = this.gradientParticles.filter(p => p.progress < 1);
  }

  // ─── Color Helpers ────────────────────────────────────────────────────────

  _weightColor(w) {
    const absW = Math.min(Math.abs(w), 2.5);
    const t = absW / 2.5;
    const alpha = 0.12 + t * 0.75;
    return w > 0
      ? `rgba(99,102,241,${alpha})`
      : `rgba(244,63,94,${alpha})`;
  }

  _activationColor(a, alpha = 1) {
    const t = Math.max(0, Math.min(1, a));
    const r = Math.round(99  + t * (251 - 99));
    const g = Math.round(102 + t * (191 - 102));
    const b = Math.round(241 + t * (36 - 241));
    return alpha < 1
      ? `rgba(${r},${g},${b},${alpha})`
      : `rgb(${r},${g},${b})`;
  }

  // ─── Draw ────────────────────────────────────────────────────────────────

  _drawConnections(layers) {
    const ctx = this.ctx;
    for (let l = 0; l < layers.length - 1; l++) {
      const from = layers[l];
      const to   = layers[l + 1];
      const W    = this.nn.weights[l];

      for (let j = 0; j < to.neurons.length; j++) {
        for (let k = 0; k < from.neurons.length; k++) {
          const w = W[j][k];
          const absW = Math.min(Math.abs(w), 2.5);
          ctx.beginPath();
          ctx.moveTo(from.neurons[k].x, from.neurons[k].y);
          ctx.lineTo(to.neurons[j].x,   to.neurons[j].y);
          ctx.strokeStyle = this._weightColor(w);
          ctx.lineWidth = 0.4 + absW * 1.4;
          ctx.stroke();
        }
      }
    }
  }

  _drawParticles() {
    const ctx = this.ctx;
    for (const p of this.gradientParticles) {
      const x = p.x + (p.tx - p.x) * p.progress;
      const y = p.y + (p.ty - p.y) * p.progress;
      ctx.beginPath();
      ctx.arc(x, y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${p.hue},90%,65%,${p.opacity})`;
      ctx.fill();
    }
  }

  _drawNeurons(layers) {
    const ctx = this.ctx;
    const pulse = Math.sin(this.pulsePhase) * 0.5 + 0.5;

    for (let l = 0; l < layers.length; l++) {
      const layer = layers[l];
      const r = this._neuronRadius(layer.neurons.length);
      const isOutput = l === layers.length - 1;
      const isInput  = l === 0;

      for (let j = 0; j < layer.neurons.length; j++) {
        const { x, y } = layer.neurons[j];
        const isSelected = this.selectedNeuron?.l === l && this.selectedNeuron?.j === j;
        const isHovered  = this.hoveredNeuron?.l  === l && this.hoveredNeuron?.j  === j;

        let aVal = 0.5;
        if (this.sampleActivations?.[l]) {
          aVal = this.sampleActivations[l][j] ?? 0.5;
        }

        // Glow
        const glowR = r * (isSelected ? 3.5 : 2.5);
        const glow = ctx.createRadialGradient(x, y, 0, x, y, glowR);
        glow.addColorStop(0, this._activationColor(aVal, isSelected ? 0.5 : 0.25));
        glow.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.beginPath();
        ctx.arc(x, y, glowR, 0, Math.PI * 2);
        ctx.fillStyle = glow;
        ctx.fill();

        // Selection ring
        if (isSelected) {
          ctx.beginPath();
          ctx.arc(x, y, r + 6, 0, Math.PI * 2);
          ctx.strokeStyle = '#fbbf24';
          ctx.lineWidth = 2;
          ctx.stroke();
        }

        // Hover ring
        if (isHovered && !isSelected) {
          ctx.beginPath();
          ctx.arc(x, y, r + 4 + pulse * 2, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(255,255,255,0.35)';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        // Output neuron pulse
        if (isOutput && !isSelected) {
          ctx.beginPath();
          ctx.arc(x, y, r + 4 + pulse * 3, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(251,191,36,${0.2 + pulse * 0.3})`;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        // Neuron body
        const bodyGrad = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, r * 0.05, x, y, r);
        bodyGrad.addColorStop(0, this._activationColor(Math.min(1, aVal + 0.25)));
        bodyGrad.addColorStop(1, this._activationColor(aVal));
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = bodyGrad;
        ctx.fill();

        // Neuron border
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.strokeStyle = isInput ? 'rgba(255,255,255,0.5)' : isSelected ? '#fbbf24' : 'rgba(255,255,255,0.18)';
        ctx.lineWidth = isSelected ? 2 : 1.5;
        ctx.stroke();

        // Activation value text
        if (r > 10) {
          ctx.fillStyle = 'rgba(255,255,255,0.92)';
          ctx.font = `bold ${Math.max(7, r * 0.52)}px Inter, sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(aVal.toFixed(2), x, y);
        }
      }
    }
  }

  _drawLayerLabels(layers) {
    const ctx = this.ctx;
    const labelY = this.canvas.height - 12;

    layers.forEach((layer, l) => {
      let label;
      if (l === 0) label = 'Input';
      else if (l === layers.length - 1) label = 'Output';
      else label = `Hidden ${l}`;

      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, layer.x, labelY);
      ctx.fillStyle = 'rgba(255,255,255,0.25)';
      ctx.font = '9px Inter, sans-serif';
      ctx.fillText(`(${layer.neurons.length})`, layer.x, labelY - 13);
    });
  }

  // ─── Main Render ────────────────────────────────────────────────────────

  render() {
    if (!this.nn) return;
    const ctx = this.ctx;
    const { width, height } = this.canvas;

    ctx.clearRect(0, 0, width, height);
    this.pulsePhase += 0.04;
    this._updateParticles();

    const layers = this._computeLayout();
    this._drawConnections(layers);
    this._drawParticles();
    this._drawNeurons(layers);
    this._drawLayerLabels(layers);

    return layers;
  }

  start() {
    const loop = () => {
      this.render();
      this.animFrame = requestAnimationFrame(loop);
    };
    if (this.animFrame) cancelAnimationFrame(this.animFrame);
    this.animFrame = requestAnimationFrame(loop);
  }

  stop() {
    if (this.animFrame) cancelAnimationFrame(this.animFrame);
  }

  // ─── Events ──────────────────────────────────────────────────────────────

  _bindEvents() {
    const getMousePos = (e) => {
      const rect = this.canvas.getBoundingClientRect();
      return {
        x: (e.clientX - rect.left) * (this.canvas.width / rect.width),
        y: (e.clientY - rect.top) * (this.canvas.height / rect.height)
      };
    };

    const findNeuron = (mx, my) => {
      if (!this.nn) return null;
      const layers = this._computeLayout();
      for (const layer of layers) {
        const r = this._neuronRadius(layer.neurons.length);
        for (const n of layer.neurons) {
          if (Math.hypot(mx - n.x, my - n.y) < r + 4) return n;
        }
      }
      return null;
    };

    this.canvas.addEventListener('mousemove', (e) => {
      const { x, y } = getMousePos(e);
      this.hoveredNeuron = findNeuron(x, y);
      this.canvas.style.cursor = this.hoveredNeuron ? 'pointer' : 'default';
    });

    this.canvas.addEventListener('click', (e) => {
      const { x, y } = getMousePos(e);
      const n = findNeuron(x, y);
      if (n) {
        this.selectedNeuron = (this.selectedNeuron?.l === n.l && this.selectedNeuron?.j === n.j) ? null : n;
        if (this.onNeuronClick && this.selectedNeuron) {
          this.onNeuronClick(n.l, n.j);
        }
      } else {
        this.selectedNeuron = null;
      }
    });
  }
}
