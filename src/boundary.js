/**
 * Decision Boundary Visualizer — v2
 * Adds: multiclass coloring (up to 6 classes), PNG export.
 */

const BOUNDARY_COLORS = [
  [99, 102, 241],    // indigo   (class 0)
  [244, 63, 94],     // rose     (class 1)
  [52, 211, 153],    // emerald  (class 2)
  [251, 191, 36],    // amber    (class 3)
  [56, 189, 248],    // sky      (class 4)
  [168, 85, 247],    // violet   (class 5)
];

class BoundaryVisualizer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.nn = null;
    this.dataset = null;
    this.numClasses = 2;
    this.problemType = 'classification';
    this._interval = null;
  }

  setNetwork(nn, numClasses = 2, problemType = 'classification') {
    this.nn = nn;
    this.numClasses = numClasses;
    this.problemType = problemType;
  }

  setDataset(dataset, numClasses = 2, problemType = 'classification') {
    this.dataset = dataset;
    this.numClasses = numClasses;
    this.problemType = problemType;
  }

  _toCanvas(x, y) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    return {
      cx: ((x + 1) / 2) * w,
      cy: (1 - (y + 1) / 2) * h
    };
  }

  // ─── Rendering ───────────────────────────────────────────────────────────

  _renderRegressionCurve() {
    if (!this.nn) return;
    const ctx = this.ctx;
    const { width, height } = this.canvas;
    
    // Draw prediction curve
    ctx.beginPath();
    const steps = 120;
    let first = true;
    for (let i = 0; i <= steps; i++) {
      const x = -1 + (i / steps) * 2;
      const pred = this.nn.predict([x]); // 1D input
      const y = pred[0]; // 1D output
      
      const { cx, cy } = this._toCanvas(x, y);
      if (first) {
        ctx.moveTo(cx, cy);
        first = false;
      } else {
        ctx.lineTo(cx, cy);
      }
    }
    
    // Glowing prediction line
    ctx.shadowColor = 'rgba(99, 102, 241, 0.4)';
    ctx.shadowBlur = 8;
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Reset shadow
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
  }

  _renderBoundaryLowRes(res = 60) {
    if (!this.nn) return;
    const { width, height } = this.canvas;
    const cellW = width / res;
    const cellH = height / res;

    for (let i = 0; i < res; i++) {
      for (let j = 0; j < res; j++) {
        const x = (i / res) * 2 - 1 + (1 / res);
        const y = 1 - ((j / res) * 2 + (1 / res));
        const pred = this.nn.predict([x, y]);

        let predClass, confidence;
        if (this.numClasses > 2) {
          predClass = pred.indexOf(Math.max(...pred));
          confidence = Math.max(...pred);
        } else {
          predClass = pred[0] > 0.5 ? 1 : 0;
          confidence = Math.abs(pred[0] - 0.5) * 2;
        }

        const [r, g, b] = BOUNDARY_COLORS[predClass % BOUNDARY_COLORS.length];
        const alpha = 0.25 + confidence * 0.5;
        this.ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
        this.ctx.fillRect(i * cellW, j * cellH, cellW + 1, cellH + 1);
      }
    }
  }

  _renderBoundaryHighRes() {
    if (!this.nn) return;
    const { width, height } = this.canvas;
    const imgData = this.ctx.createImageData(width, height);
    const data = imgData.data;

    for (let px = 0; px < width; px++) {
      for (let py = 0; py < height; py++) {
        const x = (px / width) * 2 - 1;
        const y = 1 - (py / height) * 2;
        const pred = this.nn.predict([x, y]);

        let predClass, confidence;
        if (this.numClasses > 2) {
          predClass = pred.indexOf(Math.max(...pred));
          confidence = Math.max(...pred);
        } else {
          predClass = pred[0] > 0.5 ? 1 : 0;
          confidence = Math.abs(pred[0] - 0.5) * 2;
        }

        const [r, g, b] = BOUNDARY_COLORS[predClass % BOUNDARY_COLORS.length];
        const alpha = Math.round((0.25 + confidence * 0.55) * 255);
        const idx = (py * width + px) * 4;
        data[idx]     = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = alpha;
      }
    }

    this.ctx.putImageData(imgData, 0, 0);
  }

  _renderAxes() {
    const ctx = this.ctx;
    const { width, height } = this.canvas;
    const midX = width / 2, midY = height / 2;
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(midX, 0); ctx.lineTo(midX, height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, midY); ctx.lineTo(width, midY); ctx.stroke();
    ctx.setLineDash([]);

    // Tick labels
    ctx.fillStyle = 'rgba(255,255,255,0.2)';
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ['-1', '0', '1'].forEach((lbl, i) => {
      const px = (i / 2) * width;
      ctx.fillText(lbl, px, midY - 4);
    });
  }

  _renderDataPoints() {
    if (!this.dataset) return;
    const ctx = this.ctx;
    const { width, height } = this.canvas;

    for (const pt of this.dataset) {
      const { cx, cy } = this._toCanvas(pt.x, pt.y);

      // Shadow
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0,0,0,0.35)';
      ctx.fill();

      // Point color
      let color = 'rgba(249, 115, 22, 0.9)'; // Orange for regression
      if (this.problemType === 'classification') {
        const [r, g, b] = BOUNDARY_COLORS[pt.label % BOUNDARY_COLORS.length];
        color = `rgba(${r},${g},${b},0.9)`;
      }

      ctx.beginPath();
      ctx.arc(cx, cy, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(cx, cy, 4.5, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,0.75)';
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
  }

  render(highRes = false) {
    const ctx = this.ctx;
    const { width, height } = this.canvas;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    this._renderAxes();

    if (this.nn) {
      if (this.problemType === 'regression') {
        this._renderRegressionCurve();
      } else {
        if (highRes) this._renderBoundaryHighRes();
        else this._renderBoundaryLowRes();
      }
    }

    this._renderDataPoints();
  }

  renderHighRes() {
    this.render(true);
  }

  startLiveUpdate(ms = 250) {
    if (this._interval) clearInterval(this._interval);
    this._interval = setInterval(() => this.render(false), ms);
  }

  stopLiveUpdate() {
    if (this._interval) clearInterval(this._interval);
  }

  // ─── PNG Export ───────────────────────────────────────────────────────────

  exportPNG(filename = 'decision_boundary.png') {
    this.render(true);    // high res first
    const link = document.createElement('a');
    link.download = filename;
    link.href = this.canvas.toDataURL('image/png');
    link.click();
  }
}
