/**
 * Network Topology Visualizer
 * Renders an animated, interactive diagram of the neural network
 * using Canvas 2D API, with weight magnitudes, activation glow,
 * and gradient flow animations.
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
    this.sampleActivations = null;

    this._bindEvents();
  }

  setNetwork(nn) {
    this.nn = nn;
    this.gradientParticles = [];
    this.sampleActivations = null;
  }

  setSampleActivations(activations) {
    this.sampleActivations = activations;
  }

  // ─── Layout Computation ──────────────────────────────────────────────────

  _computeLayout() {
    const { width, height } = this.canvas;
    const nn = this.nn;
    const layerCount = nn.layerSizes.length;
    const padX = 80;
    const padY = 40;
    const usableW = width - padX * 2;
    const usableH = height - padY * 2;

    const layers = [];
    for (let l = 0; l < layerCount; l++) {
      const count = nn.layerSizes[l];
      const x = padX + (l / (layerCount - 1)) * usableW;
      const neurons = [];

      for (let j = 0; j < count; j++) {
        const y = padY + ((j + 0.5) / count) * usableH;
        neurons.push({ x, y, l, j });
      }

      layers.push({ x, neurons, l });
    }

    return layers;
  }

  // ─── Particle System ─────────────────────────────────────────────────────

  spawnGradientParticles(layers) {
    if (!this.nn || !this.nn.weightGradMagnitudes) return;

    for (let l = 0; l < layers.length - 1; l++) {
      const fromLayer = layers[l];
      const toLayer = layers[l + 1];
      const gradLayer = this.nn.weightGradMagnitudes[l];

      for (let j = 0; j < toLayer.neurons.length; j++) {
        for (let k = 0; k < fromLayer.neurons.length; k++) {
          const mag = gradLayer && gradLayer[j] ? gradLayer[j][k] || 0 : 0;
          if (mag > 0.001 && Math.random() < Math.min(mag * 20, 0.4)) {
            const from = fromLayer.neurons[k];
            const to = toLayer.neurons[j];
            this.gradientParticles.push({
              x: from.x, y: from.y,
              tx: to.x, ty: to.y,
              progress: 0,
              speed: 0.015 + Math.random() * 0.02,
              size: 2 + mag * 50,
              opacity: 0.8,
              color: mag > 0.1 ? '#f97316' : '#6366f1'
            });
          }
        }
      }
    }

    // Cap particles
    if (this.gradientParticles.length > 200) {
      this.gradientParticles = this.gradientParticles.slice(-200);
    }
  }

  _updateParticles() {
    this.gradientParticles = this.gradientParticles.filter(p => p.progress < 1);
    for (const p of this.gradientParticles) {
      p.progress += p.speed;
      p.opacity = Math.max(0, 1 - p.progress);
    }
  }

  // ─── Color Helpers ───────────────────────────────────────────────────────

  _weightColor(w) {
    const absW = Math.min(Math.abs(w), 2);
    const t = absW / 2;
    if (w > 0) {
      return `rgba(99, 102, 241, ${0.2 + t * 0.8})`;   // indigo
    } else {
      return `rgba(244, 63, 94, ${0.2 + t * 0.8})`;    // rose
    }
  }

  _activationColor(a) {
    // Map activation [0,1] to a color gradient: dark → purple → gold
    const t = Math.max(0, Math.min(1, a));
    const r = Math.round(99 + t * (251 - 99));
    const g = Math.round(102 + t * (191 - 102));
    const b = Math.round(241 + t * (36 - 241));
    return `rgb(${r},${g},${b})`;
  }

  _neuronRadius(count) {
    const maxCount = Math.max(...this.nn.layerSizes);
    const base = Math.min(22, Math.max(8, 220 / maxCount));
    return base;
  }

  // ─── Draw Functions ───────────────────────────────────────────────────────

  _drawConnections(layers) {
    const ctx = this.ctx;

    for (let l = 0; l < layers.length - 1; l++) {
      const fromLayer = layers[l];
      const toLayer = layers[l + 1];
      const weightLayer = this.nn.weights[l];

      for (let j = 0; j < toLayer.neurons.length; j++) {
        for (let k = 0; k < fromLayer.neurons.length; k++) {
          const w = weightLayer[j][k];
          const from = fromLayer.neurons[k];
          const to = toLayer.neurons[j];

          const absW = Math.min(Math.abs(w), 2);
          ctx.beginPath();
          ctx.moveTo(from.x, from.y);
          ctx.lineTo(to.x, to.y);
          ctx.strokeStyle = this._weightColor(w);
          ctx.lineWidth = 0.5 + absW * 1.5;
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
      ctx.fillStyle = p.color.replace(')', `, ${p.opacity})`).replace('rgb', 'rgba');
      ctx.fill();
    }
  }

  _drawNeurons(layers) {
    const ctx = this.ctx;
    const pulse = Math.sin(this.pulsePhase) * 0.5 + 0.5;

    for (let l = 0; l < layers.length; l++) {
      const layer = layers[l];
      const count = layer.neurons.length;
      const r = this._neuronRadius(count);
      const isOutput = l === layers.length - 1;
      const isInput = l === 0;

      for (let j = 0; j < count; j++) {
        const { x, y } = layer.neurons[j];
        let activationVal = 0.5;

        if (this.sampleActivations && this.sampleActivations[l]) {
          activationVal = this.sampleActivations[l][j] ?? 0.5;
        }

        const glow = ctx.createRadialGradient(x, y, 0, x, y, r * 2.5);
        const color = this._activationColor(activationVal);
        glow.addColorStop(0, color.replace('rgb', 'rgba').replace(')', ', 0.3)'));
        glow.addColorStop(1, 'rgba(0,0,0,0)');

        // Glow
        ctx.beginPath();
        ctx.arc(x, y, r * 2.5, 0, Math.PI * 2);
        ctx.fillStyle = glow;
        ctx.fill();

        // Outer ring (pulsing for output)
        if (isOutput) {
          ctx.beginPath();
          ctx.arc(x, y, r + 4 + pulse * 3, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(251,191,36,0.4)';
          ctx.lineWidth = 2;
          ctx.stroke();
        }

        // Neuron circle
        const grad = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, r * 0.1, x, y, r);
        grad.addColorStop(0, this._activationColor(Math.min(1, activationVal + 0.3)));
        grad.addColorStop(1, this._activationColor(activationVal));

        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.strokeStyle = isInput ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.2)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Activation value label
        if (r > 12) {
          ctx.fillStyle = 'rgba(255,255,255,0.9)';
          ctx.font = `bold ${Math.max(8, r * 0.55)}px Inter, sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(activationVal.toFixed(2), x, y);
        }
      }
    }
  }

  _drawLayerLabels(layers) {
    const ctx = this.ctx;
    const labelY = this.canvas.height - 12;

    layers.forEach((layer, l) => {
      const isInput = l === 0;
      const isOutput = l === layers.length - 1;
      let label = `Hidden ${l}`;
      if (isInput) label = 'Input';
      else if (isOutput) label = 'Output';
      else label = `Hidden ${l}`;

      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.font = '11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, layer.x, labelY);
      ctx.fillText(`(${layer.neurons.length})`, layer.x, labelY - 14);
    });
  }

  // ─── Main Render Loop ────────────────────────────────────────────────────

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
    this.canvas.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (this.canvas.width / rect.width);
      const my = (e.clientY - rect.top) * (this.canvas.height / rect.height);
      this.hoveredNeuron = null;

      if (!this.nn) return;
      const layers = this._computeLayout();
      for (const layer of layers) {
        for (const n of layer.neurons) {
          const d = Math.hypot(mx - n.x, my - n.y);
          const r = this._neuronRadius(layer.neurons.length);
          if (d < r) {
            this.hoveredNeuron = n;
            break;
          }
        }
      }
    });
  }

  resize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
  }
}
