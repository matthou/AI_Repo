/**
 * Neuron Inspector
 * Renders a detailed panel when a neuron is clicked:
 *   - Incoming weight bar chart (magnitude + sign)
 *   - Bias value
 *   - Activation function curve with current z marked
 *   - Pre-activation (z) and post-activation (a) values
 */

class NeuronInspector {
  constructor(canvas, infoEl) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.infoEl = infoEl;
    this.neuron = null;   // { l, j } layer index and neuron index
    this.nn = null;
  }

  setNetwork(nn) {
    this.nn = nn;
    this.neuron = null;
    this.clear();
  }

  inspect(l, j) {
    this.neuron = { l, j };
    this.render();
  }

  clear() {
    const ctx = this.ctx;
    const { width, height } = this.canvas;
    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = 'rgba(255,255,255,0.2)';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Click a neuron to inspect it', width / 2, height / 2);

    if (this.infoEl) this.infoEl.textContent = 'No neuron selected';
  }

  render() {
    if (!this.nn || !this.neuron) { this.clear(); return; }

    const { l, j } = this.neuron;
    const { width, height } = this.canvas;
    const ctx = this.ctx;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    const isInput = l === 0;
    const isOutput = l === this.nn.numLayers - 1;

    // ─── Header ─────────────────────────────────────────────────────────────
    const layerName = isInput ? 'Input' : isOutput ? 'Output' : `Hidden ${l}`;
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`${layerName} · Neuron ${j}`, 12, 16);

    if (isInput) {
      ctx.fillStyle = 'rgba(255,255,255,0.25)';
      ctx.font = '11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Input neuron — no incoming weights', width / 2, height / 2);
      return;
    }

    // ─── Get values ──────────────────────────────────────────────────────────
    const weightLayerIdx = l - 1;
    const W = this.nn.weights[weightLayerIdx];
    const b = this.nn.biases[weightLayerIdx];

    if (!W || !W[j]) { this.clear(); return; }

    const weights = W[j];
    const bias = b[j];

    // Activation info
    const aValues = this.nn.aValues;
    const zValues = this.nn.zValues;
    const activationVal = aValues && aValues[l] ? aValues[l][j] : null;
    const zVal = zValues && zValues[weightLayerIdx] ? zValues[weightLayerIdx][j] : null;

    // ─── Weight Bar Chart ─────────────────────────────────────────────────────
    const barAreaTop = 30;
    const barAreaH = Math.floor(height * 0.45);
    const barAreaW = width - 24;
    const maxW = Math.max(...weights.map(Math.abs), 0.01);

    ctx.fillStyle = 'rgba(255,255,255,0.08)';
    ctx.fillRect(12, barAreaTop, barAreaW, barAreaH);

    const barH = Math.min(16, (barAreaH - 8) / weights.length - 2);
    const barSpacing = (barAreaH - 8) / weights.length;
    const midX = 12 + barAreaW / 2;

    // Center line
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(midX, barAreaTop + 4);
    ctx.lineTo(midX, barAreaTop + barAreaH - 4);
    ctx.stroke();

    for (let k = 0; k < weights.length; k++) {
      const w = weights[k];
      const y = barAreaTop + 4 + k * barSpacing + barSpacing / 2 - barH / 2;
      const barW = (Math.abs(w) / maxW) * (barAreaW / 2 - 8);
      const x = w >= 0 ? midX : midX - barW;

      const alpha = 0.4 + (Math.abs(w) / maxW) * 0.6;
      ctx.fillStyle = w >= 0
        ? `rgba(99,102,241,${alpha})`
        : `rgba(244,63,94,${alpha})`;
      ctx.fillRect(x, y, barW, barH);

      // Weight label
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.font = '8px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      const labelX = w >= 0 ? midX + barW + 14 : midX - barW - 14;
      ctx.fillText(w.toFixed(2), labelX, y + barH / 2 + 3);

      // From-neuron label
      ctx.fillStyle = 'rgba(255,255,255,0.25)';
      ctx.font = '8px Inter, sans-serif';
      ctx.textAlign = w >= 0 ? 'right' : 'left';
      ctx.fillText(`w${k}`, midX + (w >= 0 ? -4 : 4), y + barH / 2 + 3);
    }

    // ─── Bias bar ─────────────────────────────────────────────────────────────
    const biasY = barAreaTop + barAreaH + 6;
    ctx.fillStyle = 'rgba(255,255,255,0.12)';
    ctx.fillRect(12, biasY, barAreaW, 18);

    const biasW = (Math.abs(bias) / (maxW + 0.01)) * (barAreaW / 2 - 8);
    const biasX = bias >= 0 ? midX : midX - biasW;
    ctx.fillStyle = bias >= 0 ? 'rgba(251,191,36,0.7)' : 'rgba(251,191,36,0.4)';
    ctx.fillRect(biasX, biasY + 2, biasW, 14);

    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`bias: ${bias.toFixed(4)}`, 16, biasY + 12);

    // ─── Activation Curve ──────────────────────────────────────────────────────
    const curveTop = biasY + 24;
    const curveH = height - curveTop - 12;
    const curveW = width - 24;

    if (curveH < 30) return;

    ctx.fillStyle = 'rgba(255,255,255,0.05)';
    ctx.fillRect(12, curveTop, curveW, curveH);

    const actKey = isOutput ? this.nn.outputActivation : this.nn.activation;
    const act = Activations[actKey];

    if (act && actKey !== 'softmax') {
      const steps = Math.floor(curveW);
      const xMin = -4, xMax = 4;

      ctx.beginPath();
      let first = true;
      for (let s = 0; s <= steps; s++) {
        const xVal = xMin + (s / steps) * (xMax - xMin);
        const yVal = act.fn(xVal);
        const px = 12 + (s / steps) * curveW;
        const py = curveTop + curveH - (yVal - act.fn(xMin)) / (act.fn(xMax) - act.fn(xMin) + 1e-6) * curveH;
        if (first) { ctx.moveTo(px, py); first = false; }
        else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = '#818cf8';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Mark current z
      if (zVal !== null && isFinite(zVal)) {
        const clampedZ = Math.max(xMin, Math.min(xMax, zVal));
        const aAtZ = act.fn(zVal);
        const dotX = 12 + ((clampedZ - xMin) / (xMax - xMin)) * curveW;
        const aMin = act.fn(xMin), aMax = act.fn(xMax);
        const dotY = curveTop + curveH - (aAtZ - aMin) / (aMax - aMin + 1e-6) * curveH;

        // Crosshairs
        ctx.strokeStyle = 'rgba(251,191,36,0.4)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(dotX, curveTop);
        ctx.lineTo(dotX, curveTop + curveH);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(12, dotY);
        ctx.lineTo(12 + curveW, dotY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Dot
        ctx.beginPath();
        ctx.arc(dotX, dotY, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#fbbf24';
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Labels
      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.font = '9px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(actKey, 16, curveTop + 11);
    }

    // ─── Update info text ────────────────────────────────────────────────────
    if (this.infoEl) {
      const parts = [
        `Layer ${l} · Neuron ${j}`,
        zVal !== null ? `z = ${zVal.toFixed(4)}` : '',
        activationVal !== null ? `a = ${activationVal.toFixed(4)}` : '',
        `bias = ${bias.toFixed(4)}`
      ].filter(Boolean);
      this.infoEl.textContent = parts.join('  |  ');
    }
  }

  update() {
    if (this.neuron) this.render();
  }
}
