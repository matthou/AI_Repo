/**
 * Training Charts — v2
 * Enhanced with current-run overlay on the comparison canvas.
 */

class TrainingCharts {
  constructor(lossCanvas, accCanvas) {
    this.lossCanvas = lossCanvas;
    this.accCanvas  = accCanvas;
    this.lossCtx    = lossCanvas.getContext('2d');
    this.accCtx     = accCanvas.getContext('2d');
  }

  _renderChart(ctx, canvas, data, label, color, yMin = 0, yMax = null) {
    const { width, height } = canvas;
    const pL = 42, pR = 12, pT = 18, pB = 28;
    const plotW = width - pL - pR;
    const plotH = height - pT - pB;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    const bgG = ctx.createLinearGradient(pL, pT, pL, pT + plotH);
    bgG.addColorStop(0, 'rgba(30,41,59,0.7)');
    bgG.addColorStop(1, 'rgba(15,23,42,0.7)');
    ctx.fillStyle = bgG;
    ctx.fillRect(pL, pT, plotW, plotH);

    if (data.length < 2) {
      ctx.fillStyle = 'rgba(255,255,255,0.25)';
      ctx.font = '11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Start training to see metrics', pL + plotW / 2, pT + plotH / 2);
      ctx.textBaseline = 'alphabetic';
      return;
    }

    const maxVal = yMax !== null ? yMax : Math.max(...data) * 1.12;
    const range = maxVal - yMin || 1;

    // Grid
    for (let i = 0; i <= 4; i++) {
      const gy = pT + (i / 4) * plotH;
      const v = maxVal - (i / 4) * range;
      ctx.strokeStyle = 'rgba(255,255,255,0.055)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pL, gy); ctx.lineTo(pL + plotW, gy); ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,0.35)';
      ctx.font = '8px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(v.toFixed(2), pL - 4, gy + 3);
    }

    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.strokeRect(pL, pT, plotW, plotH);

    const pts = data.map((v, i) => ({
      x: pL + (i / (data.length - 1)) * plotW,
      y: pT + (1 - (v - yMin) / range) * plotH
    }));

    // Fill under curve
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pT + plotH);
    ctx.lineTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      const cp = (pts[i-1].x + pts[i].x) / 2;
      ctx.bezierCurveTo(cp, pts[i-1].y, cp, pts[i].y, pts[i].x, pts[i].y);
    }
    ctx.lineTo(pts[pts.length - 1].x, pT + plotH);
    ctx.closePath();
    const fillG = ctx.createLinearGradient(0, pT, 0, pT + plotH);
    fillG.addColorStop(0, color.replace('rgb(','rgba(').replace(')',',0.35)'));
    fillG.addColorStop(1, color.replace('rgb(','rgba(').replace(')',',0.02)'));
    ctx.fillStyle = fillG;
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      const cp = (pts[i-1].x + pts[i].x) / 2;
      ctx.bezierCurveTo(cp, pts[i-1].y, cp, pts[i].y, pts[i].x, pts[i].y);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();

    // End dot
    const last = pts[pts.length - 1];
    ctx.beginPath();
    ctx.arc(last.x, last.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Labels
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.font = 'bold 10px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(label, pL + 6, pT + 13);

    const lastV = data[data.length - 1];
    ctx.fillStyle = color;
    ctx.font = 'bold 12px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(lastV.toFixed(4), pL + plotW - 4, pT + 13);

    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.font = '8px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`Epoch ${data.length}`, pL + plotW / 2, height - 4);
  }

  update(nn) {
    if (!nn) return;
    this._renderChart(this.lossCtx, this.lossCanvas, nn.lossHistory, 'Train Loss', 'rgb(249,115,22)');
    this._renderChart(this.accCtx,  this.accCanvas,  nn.accHistory,  'Train Acc',  'rgb(99,102,241)', 0, 1);
  }
}
