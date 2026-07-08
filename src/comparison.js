/**
 * Run History Comparison
 * Stores past training runs and overlays their loss curves
 * on a shared canvas for direct comparison.
 */

const RUN_COLORS = [
  '#6366f1',  // indigo
  '#f97316',  // orange
  '#34d399',  // emerald
  '#fb7185',  // rose
  '#fbbf24',  // amber
  '#38bdf8',  // sky
];

class RunComparison {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.runs = [];       // [{ label, lossHistory, accHistory, color, epoch }]
    this.maxRuns = 6;
  }

  /**
   * Save the current network as a named run snapshot.
   */
  saveRun(nn, label) {
    if (this.runs.length >= this.maxRuns) {
      this.runs.shift();
    }
    const color = RUN_COLORS[this.runs.length % RUN_COLORS.length];
    this.runs.push({
      label: label || `Run ${this.runs.length + 1}`,
      lossHistory: nn.lossHistory.slice(),
      accHistory: nn.accHistory.slice(),
      color,
      epoch: nn.epoch,
      arch: nn.layerSizes.join('→'),
      activation: nn.activation,
      finalLoss: nn.lossHistory[nn.lossHistory.length - 1] ?? null,
      finalAcc: nn.accHistory[nn.accHistory.length - 1] ?? null
    });
  }

  removeRun(idx) {
    this.runs.splice(idx, 1);
  }

  clear() {
    this.runs = [];
    this._render();
  }

  // ─── Render ───────────────────────────────────────────────────────────────

  _render() {
    const ctx = this.ctx;
    const { width, height } = this.canvas;
    const padL = 36, padR = 8, padT = 16, padB = 24;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    if (this.runs.length === 0) {
      ctx.fillStyle = 'rgba(255,255,255,0.2)';
      ctx.font = '11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No saved runs yet', width / 2, height / 2);
      return;
    }

    // Find global max loss for Y scaling
    let maxLoss = 0;
    for (const run of this.runs) {
      if (run.lossHistory.length > 0) {
        maxLoss = Math.max(maxLoss, ...run.lossHistory);
      }
    }
    maxLoss = Math.max(maxLoss * 1.05, 0.01);

    // Find max epoch for X scaling
    let maxEpoch = 1;
    for (const run of this.runs) {
      maxEpoch = Math.max(maxEpoch, run.lossHistory.length);
    }

    // Grid
    for (let i = 0; i <= 4; i++) {
      const gy = padT + (i / 4) * plotH;
      const val = maxLoss * (1 - i / 4);
      ctx.strokeStyle = 'rgba(255,255,255,0.05)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, gy);
      ctx.lineTo(padL + plotW, gy);
      ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,0.3)';
      ctx.font = '8px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(val.toFixed(2), padL - 3, gy + 3);
    }

    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.strokeRect(padL, padT, plotW, plotH);

    // Draw each run's loss curve
    for (const run of this.runs) {
      if (run.lossHistory.length < 2) continue;
      const pts = run.lossHistory.map((v, i) => ({
        x: padL + (i / (maxEpoch - 1)) * plotW,
        y: padT + (1 - v / maxLoss) * plotH
      }));

      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) {
        const cpx = (pts[i - 1].x + pts[i].x) / 2;
        ctx.bezierCurveTo(cpx, pts[i - 1].y, cpx, pts[i].y, pts[i].x, pts[i].y);
      }
      ctx.strokeStyle = run.color;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // End dot
      const last = pts[pts.length - 1];
      ctx.beginPath();
      ctx.arc(last.x, last.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = run.color;
      ctx.fill();
    }

    // X axis label
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Epochs', padL + plotW / 2, height - 4);
  }

  update() {
    this._render();
  }

  /**
   * Render the run legend into a container element.
   */
  renderLegend(container) {
    container.innerHTML = '';

    if (this.runs.length === 0) {
      container.innerHTML = '<div style="color:rgba(255,255,255,0.2);font-size:11px;text-align:center;padding:8px 0;">No runs saved</div>';
      return;
    }

    this.runs.forEach((run, idx) => {
      const row = document.createElement('div');
      row.className = 'run-legend-row';
      row.innerHTML = `
        <div class="run-color-dot" style="background:${run.color}"></div>
        <div class="run-info">
          <span class="run-name">${run.label}</span>
          <span class="run-meta">[${run.arch}] · ${run.epoch}ep · loss=${(run.finalLoss ?? 0).toFixed(3)} · acc=${run.finalAcc !== null ? (run.finalAcc * 100).toFixed(1) + '%' : '—'}</span>
        </div>
        <button class="run-delete-btn" data-idx="${idx}" title="Remove run">✕</button>
      `;
      row.querySelector('.run-delete-btn').addEventListener('click', () => {
        this.removeRun(idx);
        this.update();
        this.renderLegend(container);
      });
      container.appendChild(row);
    });
  }
}
