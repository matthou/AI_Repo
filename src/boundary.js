/**
 * Decision Boundary Visualizer
 * Renders the neural network's decision boundary on a 2D canvas,
 * using a grid of sampled predictions colored by confidence.
 */

class BoundaryVisualizer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.nn = null;
    this.dataset = null;
    this.resolution = 60;   // Grid resolution
    this._imageData = null;
    this._dirty = true;
    this.animFrame = null;
  }

  setNetwork(nn) {
    this.nn = nn;
    this._dirty = true;
  }

  setDataset(dataset) {
    this.dataset = dataset;
    this._dirty = true;
  }

  markDirty() {
    this._dirty = true;
  }

  // ─── Coordinate Mapping ───────────────────────────────────────────────────

  _toCanvas(x, y) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    return {
      cx: ((x + 1) / 2) * w,
      cy: ((1 - (y + 1) / 2)) * h
    };
  }

  // ─── Boundary Rendering ───────────────────────────────────────────────────

  _renderBoundary() {
    if (!this.nn) return;
    const { width, height } = this.canvas;
    const res = this.resolution;
    const imgData = this.ctx.createImageData(width, height);
    const data = imgData.data;

    for (let px = 0; px < width; px++) {
      for (let py = 0; py < height; py++) {
        // Map pixel to [-1, 1]
        const x = (px / width) * 2 - 1;
        const y = 1 - (py / height) * 2;

        const pred = this.nn.predict([x, y])[0];

        // Color: class 0 = indigo, class 1 = rose, confidence = opacity
        const confidence = Math.abs(pred - 0.5) * 2; // 0 = uncertain, 1 = certain
        const idx = (py * width + px) * 4;

        if (pred > 0.5) {
          // Class 1: rose/orange
          data[idx]     = Math.round(244);
          data[idx + 1] = Math.round(63 + confidence * 80);
          data[idx + 2] = Math.round(94);
          data[idx + 3] = Math.round(60 + confidence * 100);
        } else {
          // Class 0: indigo/blue
          data[idx]     = Math.round(99);
          data[idx + 1] = Math.round(102);
          data[idx + 2] = Math.round(241);
          data[idx + 3] = Math.round(60 + confidence * 100);
        }
      }
    }

    this.ctx.putImageData(imgData, 0, 0);
  }

  _renderBoundaryLowRes() {
    if (!this.nn) return;
    const { width, height } = this.canvas;
    const res = this.resolution;
    const cellW = width / res;
    const cellH = height / res;

    for (let i = 0; i < res; i++) {
      for (let j = 0; j < res; j++) {
        const x = (i / res) * 2 - 1 + (1 / res);
        const y = 1 - ((j / res) * 2 + (1 / res));
        const pred = this.nn.predict([x, y])[0];
        const confidence = Math.abs(pred - 0.5) * 2;
        const alpha = Math.round(60 + confidence * 120);

        if (pred > 0.5) {
          this.ctx.fillStyle = `rgba(244, 63, 94, ${alpha / 255})`;
        } else {
          this.ctx.fillStyle = `rgba(99, 102, 241, ${alpha / 255})`;
        }
        this.ctx.fillRect(i * cellW, j * cellH, cellW + 1, cellH + 1);
      }
    }
  }

  // ─── Data Points ─────────────────────────────────────────────────────────

  _renderDataPoints() {
    if (!this.dataset) return;
    const ctx = this.ctx;
    const { width, height } = this.canvas;

    for (const pt of this.dataset) {
      const cx = ((pt.x + 1) / 2) * width;
      const cy = (1 - (pt.y + 1) / 2) * height;
      const r = 5;

      // Shadow
      ctx.beginPath();
      ctx.arc(cx, cy, r + 2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0,0,0,0.3)';
      ctx.fill();

      // Point
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = pt.label === 0 ? '#818cf8' : '#fb7185';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,0.8)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  // ─── Axes ────────────────────────────────────────────────────────────────

  _renderAxes() {
    const ctx = this.ctx;
    const { width, height } = this.canvas;
    const midX = width / 2;
    const midY = height / 2;

    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;

    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(midX, 0);
    ctx.lineTo(midX, height);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();

    ctx.setLineDash([]);
  }

  // ─── Main Render ─────────────────────────────────────────────────────────

  render(highRes = false) {
    const ctx = this.ctx;
    const { width, height } = this.canvas;

    ctx.clearRect(0, 0, width, height);

    // Dark background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    this._renderAxes();

    if (this.nn) {
      if (highRes) {
        this._renderBoundary();
      } else {
        this._renderBoundaryLowRes();
      }
    }

    this._renderDataPoints();
    this._dirty = false;
  }

  startLiveUpdate(intervalMs = 200) {
    if (this._interval) clearInterval(this._interval);
    this._interval = setInterval(() => {
      this.render(false);
    }, intervalMs);
  }

  stopLiveUpdate() {
    if (this._interval) clearInterval(this._interval);
  }

  renderHighRes() {
    this.render(true);
  }
}
