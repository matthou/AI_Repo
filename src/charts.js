/**
 * Training Charts
 * Renders live loss and accuracy curves using Canvas 2D.
 */

class TrainingCharts {
  constructor(lossCanvas, accCanvas) {
    this.lossCanvas = lossCanvas;
    this.accCanvas = accCanvas;
    this.lossCtx = lossCanvas.getContext('2d');
    this.accCtx = accCanvas.getContext('2d');
  }

  // ─── Core Rendering ───────────────────────────────────────────────────────

  _renderChart(ctx, canvas, data, label, color, yMin = 0, yMax = null, gridLines = true) {
    const { width, height } = canvas;
    const padLeft = 42;
    const padRight = 12;
    const padTop = 16;
    const padBottom = 28;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    ctx.clearRect(0, 0, width, height);

    // Background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    // Plot area background
    const bgGrad = ctx.createLinearGradient(padLeft, padTop, padLeft, padTop + plotH);
    bgGrad.addColorStop(0, 'rgba(30,41,59,0.8)');
    bgGrad.addColorStop(1, 'rgba(15,23,42,0.8)');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(padLeft, padTop, plotW, plotH);

    if (data.length < 2) {
      ctx.fillStyle = 'rgba(255,255,255,0.3)';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for training data...', padLeft + plotW / 2, padTop + plotH / 2);
      return;
    }

    // Compute Y range
    const maxVal = yMax !== null ? yMax : Math.max(...data) * 1.1;
    const minVal = yMin;
    const range = maxVal - minVal || 1;

    // Grid lines
    if (gridLines) {
      const numGrid = 4;
      for (let i = 0; i <= numGrid; i++) {
        const gy = padTop + (i / numGrid) * plotH;
        const val = maxVal - (i / numGrid) * range;

        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padLeft, gy);
        ctx.lineTo(padLeft + plotW, gy);
        ctx.stroke();

        // Y axis label
        ctx.fillStyle = 'rgba(255,255,255,0.4)';
        ctx.font = '9px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(val.toFixed(2), padLeft - 4, gy + 3);
      }
    }

    // Border
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    // Shade under curve
    const points = data.map((v, i) => ({
      x: padLeft + (i / (data.length - 1)) * plotW,
      y: padTop + (1 - (v - minVal) / range) * plotH
    }));

    ctx.beginPath();
    ctx.moveTo(points[0].x, padTop + plotH);
    ctx.lineTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      const cp1x = (points[i - 1].x + points[i].x) / 2;
      ctx.bezierCurveTo(cp1x, points[i - 1].y, cp1x, points[i].y, points[i].x, points[i].y);
    }
    ctx.lineTo(points[points.length - 1].x, padTop + plotH);
    ctx.closePath();

    const fillGrad = ctx.createLinearGradient(0, padTop, 0, padTop + plotH);
    fillGrad.addColorStop(0, color.replace(')', ', 0.3)').replace('rgb', 'rgba'));
    fillGrad.addColorStop(1, color.replace(')', ', 0.02)').replace('rgb', 'rgba'));
    ctx.fillStyle = fillGrad;
    ctx.fill();

    // Curve
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      const cp1x = (points[i - 1].x + points[i].x) / 2;
      ctx.bezierCurveTo(cp1x, points[i - 1].y, cp1x, points[i].y, points[i].x, points[i].y);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Current value dot
    const last = points[points.length - 1];
    ctx.beginPath();
    ctx.arc(last.x, last.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(last.x, last.y, 4, 0, Math.PI * 2);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Label + current value
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(label, padLeft + 6, padTop + 14);

    const lastVal = data[data.length - 1];
    ctx.fillStyle = color;
    ctx.font = 'bold 13px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(lastVal.toFixed(4), padLeft + plotW - 4, padTop + 14);

    // X axis: epoch count
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Epoch', padLeft + plotW / 2, height - 4);
    ctx.fillText('0', padLeft, height - 14);
    ctx.fillText(data.length.toString(), padLeft + plotW, height - 14);
  }

  update(nn) {
    if (!nn) return;

    this._renderChart(
      this.lossCtx, this.lossCanvas,
      nn.lossHistory, 'Loss', 'rgb(249, 115, 22)'
    );

    this._renderChart(
      this.accCtx, this.accCanvas,
      nn.accHistory, 'Accuracy', 'rgb(99, 102, 241)',
      0, 1
    );
  }
}
