/**
 * Neural Network Playground — App Controller v2
 * New: dropout, L2, multiclass, neuron inspector, run comparison,
 *      architecture presets, confusion matrix, PNG export, keyboard shortcuts,
 *      LR schedule, test evaluation.
 */

// ─── App State ────────────────────────────────────────────────────────────────

const State = {
  nn: null,
  dataset: null,
  trainData: null,
  testData: null,
  isTraining: false,
  trainInterval: null,
  currentDataset: 'spiral',
  numClasses: 2,
  problemType: 'classification',
  customPoints: [],
  activeDrawClass: 0,

  // Hyperparams
  learningRate: 0.01,
  activation: 'tanh',
  optimizer: 'adam',
  noise: 0.1,
  dataCount: 300,
  batchSize: 32,
  speed: 2,
  dropout: 0,
  l2: 0,
  lrSchedule: 'constant',

  // Architecture
  layerConfig: [2, 6, 6, 1],

  // Test metrics (updated periodically)
  testLoss: null,
  testAcc: null,
  confusionMatrix: null,

  // Effective LR (updated by scheduler)
  effectiveLR: 0.01,
};

// ─── DOM refs ─────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

let networkViz, boundaryViz, charts, inspector, comparison;

// ─── Init ─────────────────────────────────────────────────────────────────────

function init() {
  // Canvases
  const netCanvas     = $('network-canvas');
  const boundCanvas   = $('boundary-canvas');
  const lossCanvas    = $('loss-canvas');
  const accCanvas     = $('acc-canvas');
  const inspCanvas    = $('inspector-canvas');
  const compCanvas    = $('comparison-canvas');

  function resizeCanvases() {
    const vizArea = $('viz-area');
    netCanvas.width  = vizArea.clientWidth;
    netCanvas.height = vizArea.clientHeight;

    const bSize = Math.min(vizArea.clientWidth, vizArea.clientHeight) - 32;
    boundCanvas.width  = bSize;
    boundCanvas.height = bSize;

    const lossWrap = $('loss-chart-wrap');
    lossCanvas.width  = lossWrap.clientWidth;
    lossCanvas.height = lossWrap.clientHeight;

    const accWrap = $('acc-chart-wrap');
    accCanvas.width  = accWrap.clientWidth;
    accCanvas.height = accWrap.clientHeight;

    if (inspCanvas) {
      inspCanvas.width  = inspCanvas.parentElement.clientWidth;
      inspCanvas.height = inspCanvas.parentElement.clientHeight;
    }
    if (compCanvas) {
      compCanvas.width  = compCanvas.parentElement.clientWidth;
      compCanvas.height = compCanvas.parentElement.clientHeight;
    }
  }

  resizeCanvases();
  window.addEventListener('resize', () => { resizeCanvases(); });

  // Instantiate visualizers
  networkViz  = new NetworkVisualizer(netCanvas);
  boundaryViz = new BoundaryVisualizer(boundCanvas);
  charts      = new TrainingCharts(lossCanvas, accCanvas);
  inspector   = new NeuronInspector(inspCanvas, $('inspector-info'));
  comparison  = new RunComparison(compCanvas);

  // Neuron click → inspector
  networkViz.onNeuronClick = (l, j) => {
    inspector.inspect(l, j);
    // Switch to inspector tab if available
    const inspTab = document.querySelector('[data-pane="inspector"]');
    if (inspTab) {
      activateTab('inspector');
    }
  };

  buildLayerEditor();
  setupEventListeners();
  setupKeyboardShortcuts();
  generateDataset();
  buildNetwork();

  networkViz.start();
  boundaryViz.startLiveUpdate(300);
  charts.update(State.nn);
  inspector.clear();
  comparison.update();
  comparison.renderLegend($('comparison-legend'));

  // Periodic chart + confusion matrix update
  setInterval(() => {
    if (State.nn) {
      charts.update(State.nn);
      if (State.isTraining && State.nn.epoch % 10 === 0) {
        updateTestMetrics();
      }
      if (State.nn.epoch % 5 === 0) {
        inspector.update();
      }
    }
  }, 300);
}

// ─── Dataset ──────────────────────────────────────────────────────────────────

function generateDataset(clearCustom = true) {
  if (clearCustom) State.customPoints = [];

  const key = State.currentDataset;
  const def = Datasets[key];
  const problemType = def.problemType || 'classification';
  State.problemType = problemType;
  State.numClasses = def.classes;

  // Set UI labels
  if (problemType === 'regression') {
    $('lbl-train-acc').textContent = 'Train R²';
    $('lbl-test-acc').textContent = 'Test R²';
    $('confusion-matrix-section').style.display = 'none';
    
    // Hide drawing buttons for classes
    $('btn-draw-c0').textContent = 'Add Point';
    $('btn-draw-c1').style.display = 'none';
    $('btn-draw-c2').style.display = 'none';
  } else {
    $('lbl-train-acc').textContent = 'Train Acc';
    $('lbl-test-acc').textContent = 'Accuracy';
    $('confusion-matrix-section').style.display = 'block';
    
    $('btn-draw-c0').textContent = 'Class 0';
    $('btn-draw-c1').style.display = 'inline-block';
    if (State.numClasses > 2) {
      $('btn-draw-c2').style.display = 'inline-block';
    } else {
      $('btn-draw-c2').style.display = 'none';
    }
  }

  refreshDatasetFromPoints();
}

function refreshDatasetFromPoints() {
  const key = State.currentDataset;
  const def = Datasets[key];
  const problemType = State.problemType;
  
  // Generate base dataset
  const raw = def.generate(State.dataCount, State.noise);
  
  // Combine with custom points
  const combined = raw.concat(State.customPoints);
  
  const prepared = prepareDataset(combined, State.numClasses, problemType);
  const split = trainTestSplit(prepared, 0.8);
  
  State.dataset   = prepared;
  State.trainData = split.train;
  State.testData  = split.test;
  
  boundaryViz.setDataset(prepared, State.numClasses, problemType);
  boundaryViz.render(false);
  renderDatasetPreview(combined);
  
  if (State.trainData?.length > 0) {
    const sample = State.trainData[0];
    const acts = State.nn ? State.nn.getActivations(sample.input) : null;
    if (acts && networkViz) networkViz.setSampleActivations(acts);
  }
  
  updateStats();
  updateWeightsDisplay();
  updateTestMetrics();
}

function renderDatasetPreview(rawData) {
  const canvas = $('dataset-preview');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, w, h);

  const colors = ['#818cf8','#fb7185','#34d399','#fbbf24','#38bdf8','#a855f7'];
  for (const pt of rawData) {
    const cx = ((pt.x + 1) / 2) * w;
    const cy = (1 - (pt.y + 1) / 2) * h;
    ctx.beginPath();
    ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = State.problemType === 'regression' ? '#f97316' : colors[pt.label % colors.length];
    ctx.fill();
  }
}

// ─── Network ──────────────────────────────────────────────────────────────────

function buildNetwork() {
  if (State.isTraining) stopTraining();

  const problemType = State.problemType;
  const inputSize = problemType === 'regression' ? 1 : 2;
  const outputSize = problemType === 'regression' ? 1 : (State.numClasses > 2 ? State.numClasses : 1);
  
  const layerConfig = State.layerConfig.slice();
  layerConfig[0] = inputSize;
  layerConfig[layerConfig.length - 1] = outputSize;

  const outputAct = problemType === 'regression' ? 'linear' : (State.numClasses > 2 ? 'softmax' : 'sigmoid');

  State.nn = new NeuralNetwork(
    layerConfig,
    State.activation,
    outputAct,
    { dropout: State.dropout, l2: State.l2 }
  );
  State.nn.lrSchedule  = State.lrSchedule;
  State.nn.totalEpochs = 1000;

  networkViz.setNetwork(State.nn);
  boundaryViz.setNetwork(State.nn, State.numClasses, problemType);
  inspector.setNetwork(State.nn);

  if (State.trainData?.length > 0) {
    const sample = State.trainData[0];
    const acts = State.nn.getActivations(sample.input);
    networkViz.setSampleActivations(acts);
  }

  updateStats();
  updateWeightsDisplay();
  updateConfusionMatrix(null);
  showToast(`🔨 Built [${layerConfig.join('→')}] · ${outputAct}`);
}

function resetNetwork() {
  if (!State.nn) return;
  State.nn.initWeights();
  State.nn.lrSchedule  = State.lrSchedule;
  State.nn.totalEpochs = 1000;
  updateStats();
  updateWeightsDisplay();
  charts.update(State.nn);
  showToast('🔄 Weights reset');
}

// ─── Architecture Presets ─────────────────────────────────────────────────────

const PRESETS = {
  tiny:   { layers: [2, 4, 1],         label: 'Tiny' },
  small:  { layers: [2, 8, 8, 1],      label: 'Small' },
  medium: { layers: [2, 16, 16, 1],    label: 'Medium' },
  deep:   { layers: [2, 8, 8, 8, 8, 1],label: 'Deep' },
  wide:   { layers: [2, 32, 32, 1],    label: 'Wide' },
};

function applyPreset(key) {
  const preset = PRESETS[key];
  if (!preset) return;
  State.layerConfig = preset.layers.slice();
  buildLayerEditor();
  buildNetwork();
  showToast(`📐 Preset: ${preset.label} [${preset.layers.join('→')}]`);
}

// ─── Training ─────────────────────────────────────────────────────────────────

function startTraining() {
  if (State.isTraining || !State.nn || !State.trainData) return;
  State.isTraining = true;
  $('btn-train').disabled = true;
  $('btn-stop').disabled  = false;
  $('status-text').textContent = 'Training';
  $('status-badge').classList.add('running');

  let tickCount = 0;

  State.trainInterval = setInterval(() => {
    for (let e = 0; e < State.speed; e++) {
      const shuffled   = State.trainData.slice().sort(() => Math.random() - 0.5);
      const batchSize  = Math.min(State.batchSize, shuffled.length);
      const batch      = shuffled.slice(0, batchSize);
      const inputs     = batch.map(d => d.input);
      const targets    = batch.map(d => d.target);
      const result     = State.nn.trainBatch(inputs, targets, State.learningRate, State.optimizer);
      State.effectiveLR = result.effectiveLR;
    }

    // Update activations sample
    if (State.trainData.length > 0) {
      const sample = State.trainData[Math.floor(Math.random() * Math.min(5, State.trainData.length))];
      const acts = State.nn.getActivations(sample.input);
      networkViz.setSampleActivations(acts);
    }

    // Gradient particles
    networkViz.spawnGradientParticles(networkViz._computeLayout());

    tickCount++;
    updateStats();

    if (tickCount % 3 === 0) updateWeightsDisplay();
    if (tickCount % 10 === 0) updateTestMetrics();

  }, 100);
}

function stopTraining() {
  if (State.trainInterval) clearInterval(State.trainInterval);
  State.trainInterval = null;
  State.isTraining    = false;
  $('btn-train').disabled = false;
  $('btn-stop').disabled  = true;
  $('status-text').textContent = 'Idle';
  $('status-badge').classList.remove('running');

  updateTestMetrics();
  setTimeout(() => boundaryViz.renderHighRes(), 150);
  showToast(`⏹️ Stopped at epoch ${State.nn.epoch}`);
}

// ─── Stats & Metrics ──────────────────────────────────────────────────────────

function updateStats() {
  if (!State.nn) return;
  const { epoch, lossHistory, accHistory } = State.nn;

  $('stat-epoch').textContent = epoch;

  const loss = lossHistory.at(-1) ?? 0;
  const acc  = accHistory.at(-1)  ?? 0;
  $('stat-loss').textContent = loss.toFixed(4);
  
  if (State.problemType === 'regression') {
    $('stat-acc').textContent  = acc.toFixed(3);
  } else {
    $('stat-acc').textContent  = (acc * 100).toFixed(1) + '%';
  }
  
  $('stat-params').textContent = State.nn.paramCount();

  const lr = State.effectiveLR ?? State.learningRate;
  $('stat-lr').textContent = lr < 0.001 ? lr.toExponential(1) : lr.toFixed(4);
}

function updateTestMetrics() {
  if (!State.nn || !State.testData || State.testData.length === 0) return;
  const inputs  = State.testData.map(d => d.input);
  const targets = State.testData.map(d => d.target);
  const result  = State.nn.evaluate(inputs, targets);

  State.testLoss = result.loss;
  State.testAcc  = result.acc;
  State.confusionMatrix = result.confusionMatrix;

  const el = $('stat-test-acc');
  if (el) {
    if (State.problemType === 'regression') {
      el.textContent = result.acc.toFixed(3);
    } else {
      el.textContent = (result.acc * 100).toFixed(1) + '%';
    }
  }
  const elLoss = $('stat-test-loss');
  if (elLoss) elLoss.textContent = result.loss.toFixed(4);

  updateConfusionMatrix(result.confusionMatrix);
}

function updateConfusionMatrix(matrix) {
  const container = $('confusion-matrix');
  if (!container) return;
  container.innerHTML = '';

  if (State.problemType === 'regression') {
    container.innerHTML = '<div style="color:var(--text-secondary);font-size:11px;text-align:center;padding:12px 0;">Not applicable for regression</div>';
    return;
  }

  if (!matrix) {
    container.innerHTML = '<div style="color:rgba(255,255,255,0.2);font-size:11px;text-align:center;padding:12px 0;">Train first to see confusion matrix</div>';
    return;
  }

  const n = matrix.length;
  if (n > 6) {
    container.innerHTML = '<div style="color:rgba(255,255,255,0.3);font-size:11px;">Matrix too large to display</div>';
    return;
  }

  const maxVal = Math.max(...matrix.flat(), 1);
  const classColors = ['#818cf8','#fb7185','#34d399','#fbbf24','#38bdf8','#a855f7'];

  // Header row
  const header = document.createElement('div');
  header.className = 'cm-row';
  header.appendChild(Object.assign(document.createElement('div'), { className: 'cm-axis-label', textContent: 'A\\P' }));
  for (let p = 0; p < n; p++) {
    const h = document.createElement('div');
    h.className = 'cm-col-header';
    h.textContent = p;
    h.style.color = classColors[p];
    header.appendChild(h);
  }
  container.appendChild(header);

  for (let a = 0; a < n; a++) {
    const row = document.createElement('div');
    row.className = 'cm-row';

    const rowLabel = document.createElement('div');
    rowLabel.className = 'cm-row-label';
    rowLabel.textContent = a;
    rowLabel.style.color = classColors[a];
    row.appendChild(rowLabel);

    for (let p = 0; p < n; p++) {
      const cell = document.createElement('div');
      cell.className = 'cm-cell';
      const v = matrix[a][p];
      const intensity = v / maxVal;
      const isDiag = a === p;

      cell.style.background = isDiag
        ? `rgba(99,102,241,${0.15 + intensity * 0.75})`
        : `rgba(244,63,94,${intensity * 0.6})`;
      cell.style.color = intensity > 0.3 ? '#fff' : 'rgba(255,255,255,0.5)';
      cell.textContent = v;
      cell.title = `True: ${a}, Pred: ${p} = ${v}`;
      row.appendChild(cell);
    }
    container.appendChild(row);
  }

  // Legend
  const leg = document.createElement('div');
  leg.className = 'cm-legend';
  leg.innerHTML = '<span style="color:#818cf8">■</span> Correct &nbsp; <span style="color:#fb7185">■</span> Wrong';
  container.appendChild(leg);
}

// ─── Weights Display ──────────────────────────────────────────────────────────

function updateWeightsDisplay() {
  if (!State.nn) return;
  const container = $('weights-display');
  if (!container) return;
  container.innerHTML = '';

  for (let l = 0; l < State.nn.weights.length; l++) {
    const label = document.createElement('div');
    label.className = 'weight-layer-label';
    label.textContent = `L${l + 1} weights`;
    container.appendChild(label);

    const matrix = document.createElement('div');
    matrix.className = 'weight-matrix';

    const W = State.nn.weights[l];
    const maxRows = Math.min(W.length, 8);
    const maxCols = Math.min(W[0].length, 8);

    for (let j = 0; j < maxRows; j++) {
      for (let k = 0; k < maxCols; k++) {
        const w = W[j][k];
        const cell = document.createElement('div');
        cell.className = 'weight-cell';
        cell.title = `w[${j}][${k}] = ${w.toFixed(4)}`;
        const t = Math.min(Math.abs(w), 2) / 2;
        const alpha = 0.25 + t * 0.7;
        cell.style.background = w > 0
          ? `rgba(99,102,241,${alpha})`
          : `rgba(244,63,94,${alpha})`;
        cell.textContent = w.toFixed(1);
        matrix.appendChild(cell);
      }
    }
    container.appendChild(matrix);
  }
}

// ─── Layer Editor ─────────────────────────────────────────────────────────────

function buildLayerEditor() {
  const container = $('layers-editor');
  container.innerHTML = '';
  const config = State.layerConfig;

  config.forEach((count, idx) => {
    const isInput  = idx === 0;
    const isOutput = idx === config.length - 1;

    const row = document.createElement('div');
    row.className = 'layer-row';

    const typeSpan = document.createElement('span');
    typeSpan.className = `layer-type ${isInput ? 'input' : isOutput ? 'output' : 'hidden'}`;
    typeSpan.textContent = isInput ? 'In' : isOutput ? 'Out' : `H${idx}`;

    const countInput = document.createElement('input');
    countInput.type = 'number';
    countInput.className = 'layer-count-input';
    countInput.value = count;
    countInput.min = 1;
    countInput.max = 32;
    countInput.disabled = isInput || isOutput;
    if (!isInput && !isOutput) {
      countInput.addEventListener('change', () => {
        State.layerConfig[idx] = Math.max(1, Math.min(32, parseInt(countInput.value) || 1));
        countInput.value = State.layerConfig[idx];
      });
    }

    const removeBtn = document.createElement('button');
    removeBtn.className = 'layer-remove-btn';
    removeBtn.innerHTML = '✕';
    removeBtn.disabled = !(!isInput && !isOutput);
    if (!isInput && !isOutput) {
      removeBtn.addEventListener('click', () => {
        State.layerConfig.splice(idx, 1);
        buildLayerEditor();
      });
    }

    row.appendChild(typeSpan);
    row.appendChild(countInput);
    row.appendChild(removeBtn);
    container.appendChild(row);
  });

  const addBtn = document.createElement('button');
  addBtn.className = 'add-layer-btn';
  addBtn.innerHTML = '＋ Add Hidden Layer';
  addBtn.addEventListener('click', () => {
    State.layerConfig.splice(State.layerConfig.length - 1, 0, 8);
    buildLayerEditor();
  });
  container.appendChild(addBtn);
}

// ─── Tab System ───────────────────────────────────────────────────────────────

function activateTab(paneName) {
  document.querySelectorAll('.viz-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.viz-pane').forEach(p => p.classList.remove('active'));
  const tab  = document.querySelector(`[data-pane="${paneName}"]`);
  const pane = $(`pane-${paneName}`);
  if (tab)  tab.classList.add('active');
  if (pane) pane.classList.add('active');
}

// ─── Event Listeners ──────────────────────────────────────────────────────────

function setupEventListeners() {
  $('btn-train').addEventListener('click', startTraining);
  $('btn-stop').addEventListener('click', stopTraining);
  $('btn-rebuild').addEventListener('click', buildNetwork);
  $('btn-reset').addEventListener('click', resetNetwork);

  $('dataset-select').addEventListener('change', e => {
    State.currentDataset = e.target.value;
    generateDataset();
    buildNetwork();
  });

  $('noise-slider').addEventListener('input', e => {
    State.noise = parseFloat(e.target.value);
    $('noise-val').textContent = State.noise.toFixed(2);
    generateDataset(false /* keep custom points */);
    buildNetwork();
  });

  $('data-count').addEventListener('change', e => {
    State.dataCount = parseInt(e.target.value);
    generateDataset();
  });

  $('activation-select').addEventListener('change', e => {
    State.activation = e.target.value;
  });

  $('lr-slider').addEventListener('input', e => {
    State.learningRate = Math.pow(10, parseFloat(e.target.value));
    $('lr-val').textContent = State.learningRate.toExponential(0);
  });

  $('optimizer-select').addEventListener('change', e => {
    State.optimizer = e.target.value;
  });

  $('batch-size').addEventListener('change', e => {
    State.batchSize = parseInt(e.target.value);
  });

  $('dropout-slider').addEventListener('input', e => {
    State.dropout = parseFloat(e.target.value);
    $('dropout-val').textContent = (State.dropout * 100).toFixed(0) + '%';
    if (State.nn) State.nn.dropout = State.dropout;
  });

  $('l2-slider').addEventListener('input', e => {
    State.l2 = parseFloat(e.target.value);
    $('l2-val').textContent = State.l2.toExponential(0);
    if (State.nn) State.nn.l2 = State.l2;
  });

  $('lr-schedule-select').addEventListener('change', e => {
    State.lrSchedule = e.target.value;
    if (State.nn) State.nn.lrSchedule = State.lrSchedule;
  });

  // Speed buttons
  document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      State.speed = parseInt(btn.dataset.speed);
    });
  });

  // Viz tabs
  document.querySelectorAll('.viz-tab').forEach(tab => {
    tab.addEventListener('click', () => activateTab(tab.dataset.pane));
  });

  // Preset buttons
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => applyPreset(btn.dataset.preset));
  });

  // Export / Import
  $('btn-export').addEventListener('click', exportModel);
  $('btn-import').addEventListener('click', () => $('import-file').click());
  $('import-file').addEventListener('change', importModel);

  // Export boundary PNG
  $('btn-export-boundary').addEventListener('click', () => {
    boundaryViz.exportPNG(`boundary_epoch${State.nn?.epoch ?? 0}.png`);
    showToast('🖼️ Boundary exported as PNG');
  });

  // Run comparison
  $('btn-save-run').addEventListener('click', () => {
    if (!State.nn || State.nn.epoch === 0) {
      showToast('⚠️ Train the network first');
      return;
    }
    const label = `[${State.layerConfig.join('→')}] ${State.activation}`;
    comparison.saveRun(State.nn, label);
    comparison.update();
    comparison.renderLegend($('comparison-legend'));
    showToast('📌 Run saved to comparison');
  });

  $('btn-clear-runs').addEventListener('click', () => {
    comparison.clear();
    comparison.renderLegend($('comparison-legend'));
    showToast('🗑️ Comparison runs cleared');
  });

  // Interactive custom point drawing selectors
  const drawBtns = [$('btn-draw-c0'), $('btn-draw-c1'), $('btn-draw-c2')];
  drawBtns.forEach((btn, idx) => {
    btn.addEventListener('click', () => {
      drawBtns.forEach(b => {
        b.classList.remove('active');
        b.style.borderColor = '';
        b.style.color = '';
      });
      btn.classList.add('active');
      btn.style.borderColor = 'var(--indigo-500)';
      btn.style.color = 'var(--indigo-400)';
      State.activeDrawClass = idx;
    });
  });

  // Click on decision boundary canvas to place point
  $('boundary-canvas').addEventListener('click', e => {
    if (State.isTraining) return;

    const rect = $('boundary-canvas').getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    const x = (px / rect.width) * 2 - 1;
    const y = 1 - (py / rect.height) * 2;

    const newPt = { x, y, label: State.problemType === 'regression' ? 0 : State.activeDrawClass };
    State.customPoints.push(newPt);

    refreshDatasetFromPoints();
    showToast(`📍 Custom point placed at (${x.toFixed(2)}, ${y.toFixed(2)})`);
  });
}

// ─── Keyboard Shortcuts ───────────────────────────────────────────────────────

function setupKeyboardShortcuts() {
  document.addEventListener('keydown', e => {
    // Don't trigger when typing in inputs
    if (['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)) return;

    switch (e.code) {
      case 'Space':
        e.preventDefault();
        State.isTraining ? stopTraining() : startTraining();
        break;
      case 'KeyR':
        resetNetwork();
        break;
      case 'KeyG':
        generateDataset();
        buildNetwork();
        showToast('🎲 Data regenerated');
        break;
      case 'KeyB':
        activateTab('boundary');
        break;
      case 'KeyN':
        activateTab('network');
        break;
      case 'KeyI':
        activateTab('inspector');
        break;
      case 'KeyC':
        activateTab('comparison');
        break;
      case 'Digit1': applyPreset('tiny');   break;
      case 'Digit2': applyPreset('small');  break;
      case 'Digit3': applyPreset('medium'); break;
      case 'Digit4': applyPreset('deep');   break;
      case 'Digit5': applyPreset('wide');   break;
    }
  });
}

// ─── Export / Import ──────────────────────────────────────────────────────────

function exportModel() {
  if (!State.nn) return;
  const payload = {
    ...State.nn.toJSON(),
    meta: {
      exportedAt: new Date().toISOString(),
      datasetKey: State.currentDataset,
      noise: State.noise
    }
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url;
  a.download = `nn_model_e${State.nn.epoch}.json`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('💾 Model exported!');
}

function importModel(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    try {
      const data = JSON.parse(ev.target.result);
      State.nn = NeuralNetwork.fromJSON(data);
      State.layerConfig = data.layerSizes.slice();
      State.activation  = data.activation;
      State.dropout = data.dropout || 0;
      State.l2      = data.l2 || 0;

      networkViz.setNetwork(State.nn);
      boundaryViz.setNetwork(State.nn, State.numClasses);
      inspector.setNetwork(State.nn);
      buildLayerEditor();
      updateStats();
      updateWeightsDisplay();
      charts.update(State.nn);
      showToast(`📂 Imported! Epoch: ${State.nn.epoch}`);
    } catch {
      showToast('❌ Failed to import model');
    }
  };
  reader.readAsText(file);
  e.target.value = '';
}

// ─── Toast ────────────────────────────────────────────────────────────────────

let toastTimeout;
function showToast(msg) {
  const toast = $('toast');
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => toast.classList.remove('show'), 2800);
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', init);
