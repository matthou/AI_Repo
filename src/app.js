/**
 * Neural Network Playground — Main Application Controller
 * Orchestrates training loop, UI interactions, and visualization updates.
 */

// ─── App State ────────────────────────────────────────────────────────────────

const State = {
  nn: null,
  dataset: null,
  trainData: null,
  testData: null,
  isTraining: false,
  trainInterval: null,
  epochsPerTick: 1,
  currentDataset: 'spiral',
  layerConfig: [2, 6, 6, 1],       // input, hidden..., output
  learningRate: 0.01,
  activation: 'tanh',
  optimizer: 'adam',
  noise: 0.1,
  dataCount: 300,
  batchSize: 32,
  speed: 2,                          // epochs per 100ms tick
};

// ─── DOM Refs ─────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

let networkViz, boundaryViz, charts;

// ─── Initialization ───────────────────────────────────────────────────────────

function init() {
  // Canvas setup
  const netCanvas = $('network-canvas');
  const boundCanvas = $('boundary-canvas');
  const lossCanvas = $('loss-canvas');
  const accCanvas = $('acc-canvas');

  function resizeCanvases() {
    const vizArea = $('viz-area');
    netCanvas.width = vizArea.clientWidth;
    netCanvas.height = vizArea.clientHeight;

    const bSize = Math.min(vizArea.clientWidth, vizArea.clientHeight) - 32;
    boundCanvas.width = bSize;
    boundCanvas.height = bSize;

    const lossWrap = $('loss-chart-wrap');
    lossCanvas.width = lossWrap.clientWidth;
    lossCanvas.height = lossWrap.clientHeight;

    const accWrap = $('acc-chart-wrap');
    accCanvas.width = accWrap.clientWidth;
    accCanvas.height = accWrap.clientHeight;
  }

  resizeCanvases();
  window.addEventListener('resize', () => {
    resizeCanvases();
    if (networkViz) networkViz.render();
  });

  networkViz = new NetworkVisualizer(netCanvas);
  boundaryViz = new BoundaryVisualizer(boundCanvas);
  charts = new TrainingCharts(lossCanvas, accCanvas);

  buildLayerEditor();
  setupEventListeners();
  generateDataset();
  buildNetwork();
  networkViz.start();
  boundaryViz.startLiveUpdate(250);

  updateChartsLoop();
}

// ─── Dataset ──────────────────────────────────────────────────────────────────

function generateDataset() {
  const key = State.currentDataset;
  const raw = Datasets[key].generate(State.dataCount, State.noise);
  const prepared = prepareDataset(raw);
  const split = trainTestSplit(prepared, 0.8);
  State.dataset = prepared;
  State.trainData = split.train;
  State.testData = split.test;

  boundaryViz.setDataset(prepared);
  boundaryViz.render(false);

  // Draw preview
  renderDatasetPreview(raw);

  showToast(`📊 Generated ${prepared.length} ${Datasets[key].label} samples`);
}

function renderDatasetPreview(rawData) {
  const canvas = $('dataset-preview');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, w, h);

  for (const pt of rawData) {
    const cx = ((pt.x + 1) / 2) * w;
    const cy = (1 - (pt.y + 1) / 2) * h;
    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, Math.PI * 2);
    ctx.fillStyle = pt.label === 0 ? '#818cf8' : '#fb7185';
    ctx.fill();
  }
}

// ─── Network ──────────────────────────────────────────────────────────────────

function buildNetwork() {
  if (State.isTraining) stopTraining();

  State.nn = new NeuralNetwork(
    State.layerConfig.slice(),
    State.activation,
    'sigmoid'
  );

  networkViz.setNetwork(State.nn);
  boundaryViz.setNetwork(State.nn);

  // Run a sample activation
  if (State.trainData && State.trainData.length > 0) {
    const sample = State.trainData[0];
    const acts = State.nn.getActivations(sample.input);
    networkViz.setSampleActivations(acts);
  }

  updateStats();
  updateWeightsDisplay();
  showToast(`🔨 Network rebuilt: [${State.layerConfig.join(' → ')}]`);
}

function resetNetwork() {
  if (State.nn) {
    State.nn.initWeights();
    updateStats();
    updateWeightsDisplay();
    showToast('🔄 Weights reset');
  }
}

// ─── Training ─────────────────────────────────────────────────────────────────

function startTraining() {
  if (State.isTraining) return;
  if (!State.nn || !State.trainData) return;

  State.isTraining = true;
  $('btn-train').disabled = true;
  $('btn-stop').disabled = false;
  $('status-text').textContent = 'Training';
  $('status-badge').classList.add('running');

  let ticksPerUpdate = 0;

  State.trainInterval = setInterval(() => {
    // Run N epochs per tick
    for (let e = 0; e < State.speed; e++) {
      // Mini-batch
      const shuffled = State.trainData.slice().sort(() => Math.random() - 0.5);
      const batchSize = Math.min(State.batchSize, shuffled.length);
      const batch = shuffled.slice(0, batchSize);

      const inputs = batch.map(d => d.input);
      const targets = batch.map(d => d.target);

      State.nn.trainBatch(inputs, targets, State.learningRate, State.optimizer);
    }

    // Update sample activations (for visualizer)
    if (State.trainData.length > 0) {
      const sample = State.trainData[Math.floor(Math.random() * Math.min(5, State.trainData.length))];
      const acts = State.nn.getActivations(sample.input);
      networkViz.setSampleActivations(acts);
    }

    // Spawn gradient particles
    const layers = networkViz._computeLayout();
    networkViz.spawnGradientParticles(layers);

    updateStats();

    ticksPerUpdate++;
    if (ticksPerUpdate >= 3) {
      updateWeightsDisplay();
      ticksPerUpdate = 0;
    }
  }, 100);
}

function stopTraining() {
  if (State.trainInterval) clearInterval(State.trainInterval);
  State.trainInterval = null;
  State.isTraining = false;
  $('btn-train').disabled = false;
  $('btn-stop').disabled = true;
  $('status-text').textContent = 'Idle';
  $('status-badge').classList.remove('running');

  // High-res boundary render when stopped
  setTimeout(() => boundaryViz.renderHighRes(), 100);
  showToast(`⏹️ Training stopped at epoch ${State.nn.epoch}`);
}

// ─── Stats ────────────────────────────────────────────────────────────────────

function updateStats() {
  if (!State.nn) return;
  const { epoch, lossHistory, accHistory } = State.nn;

  $('stat-epoch').textContent = epoch;

  const loss = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1] : 0;
  const acc = accHistory.length > 0 ? accHistory[accHistory.length - 1] : 0;

  $('stat-loss').textContent = loss.toFixed(4);
  $('stat-acc').textContent = (acc * 100).toFixed(1) + '%';

  // Parameter count
  let params = 0;
  for (let l = 0; l < State.nn.weights.length; l++) {
    params += State.nn.weights[l].length * State.nn.weights[l][0].length;
    params += State.nn.biases[l].length;
  }
  $('stat-params').textContent = params;
}

function updateChartsLoop() {
  setInterval(() => {
    if (State.nn && State.isTraining) {
      charts.update(State.nn);
    }
  }, 300);
}

// ─── Weights Display ──────────────────────────────────────────────────────────

function updateWeightsDisplay() {
  if (!State.nn) return;
  const container = $('weights-display');
  container.innerHTML = '';

  for (let l = 0; l < State.nn.weights.length; l++) {
    const label = document.createElement('div');
    label.className = 'weight-layer-label';
    label.textContent = `Layer ${l + 1} weights`;
    container.appendChild(label);

    const matrix = document.createElement('div');
    matrix.className = 'weight-matrix';

    const W = State.nn.weights[l];
    // Limit display to 8x8
    const maxRows = Math.min(W.length, 8);
    const maxCols = Math.min(W[0].length, 8);

    for (let j = 0; j < maxRows; j++) {
      for (let k = 0; k < maxCols; k++) {
        const w = W[j][k];
        const cell = document.createElement('div');
        cell.className = 'weight-cell';
        cell.title = w.toFixed(4);

        const absW = Math.min(Math.abs(w), 2);
        const t = absW / 2;
        if (w > 0) {
          const a = Math.round(60 + t * 180);
          cell.style.background = `rgba(99,102,241,${(60 + t * 180) / 255})`;
        } else {
          cell.style.background = `rgba(244,63,94,${(60 + t * 180) / 255})`;
        }

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
    const isInput = idx === 0;
    const isOutput = idx === config.length - 1;
    const isHidden = !isInput && !isOutput;

    const row = document.createElement('div');
    row.className = 'layer-row';
    row.dataset.idx = idx;

    const typeSpan = document.createElement('span');
    typeSpan.className = `layer-type ${isInput ? 'input' : isOutput ? 'output' : 'hidden'}`;
    typeSpan.textContent = isInput ? 'Input' : isOutput ? 'Output' : `H${idx}`;

    const countInput = document.createElement('input');
    countInput.type = 'number';
    countInput.className = 'layer-count-input';
    countInput.value = count;
    countInput.min = 1;
    countInput.max = 16;
    countInput.disabled = isInput; // input is always 2 (x, y)

    if (!isInput) {
      countInput.addEventListener('change', () => {
        const val = Math.max(1, Math.min(16, parseInt(countInput.value) || 1));
        countInput.value = val;
        State.layerConfig[idx] = val;
      });
    }

    const removeBtn = document.createElement('button');
    removeBtn.className = 'layer-remove-btn';
    removeBtn.innerHTML = '✕';
    removeBtn.disabled = !isHidden;
    removeBtn.title = isHidden ? 'Remove layer' : '';
    if (isHidden) {
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

  // Add layer button
  const addBtn = document.createElement('button');
  addBtn.className = 'add-layer-btn';
  addBtn.innerHTML = '＋ Add Hidden Layer';
  addBtn.addEventListener('click', () => {
    // Insert before output layer
    State.layerConfig.splice(State.layerConfig.length - 1, 0, 4);
    buildLayerEditor();
  });
  container.appendChild(addBtn);
}

// ─── Event Listeners ──────────────────────────────────────────────────────────

function setupEventListeners() {
  // Train / Stop buttons
  $('btn-train').addEventListener('click', startTraining);
  $('btn-stop').addEventListener('click', stopTraining);
  $('btn-rebuild').addEventListener('click', buildNetwork);
  $('btn-reset').addEventListener('click', resetNetwork);

  // Dataset selector
  $('dataset-select').addEventListener('change', e => {
    State.currentDataset = e.target.value;
    generateDataset();
    buildNetwork();
  });

  // Noise
  $('noise-slider').addEventListener('input', e => {
    State.noise = parseFloat(e.target.value);
    $('noise-val').textContent = State.noise.toFixed(2);
    generateDataset();
    buildNetwork();
  });

  // Data count
  $('data-count').addEventListener('change', e => {
    State.dataCount = parseInt(e.target.value);
    generateDataset();
  });

  // Activation
  $('activation-select').addEventListener('change', e => {
    State.activation = e.target.value;
  });

  // Learning rate
  $('lr-slider').addEventListener('input', e => {
    const exp = parseFloat(e.target.value);
    State.learningRate = Math.pow(10, exp);
    $('lr-val').textContent = State.learningRate.toExponential(0);
  });

  // Optimizer
  $('optimizer-select').addEventListener('change', e => {
    State.optimizer = e.target.value;
  });

  // Batch size
  $('batch-size').addEventListener('change', e => {
    State.batchSize = parseInt(e.target.value);
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
    tab.addEventListener('click', () => {
      document.querySelectorAll('.viz-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.viz-pane').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      $(`pane-${tab.dataset.pane}`).classList.add('active');
    });
  });

  // Export / Import
  $('btn-export').addEventListener('click', exportModel);
  $('btn-import').addEventListener('click', () => $('import-file').click());
  $('import-file').addEventListener('change', importModel);
}

// ─── Export / Import ──────────────────────────────────────────────────────────

function exportModel() {
  if (!State.nn) return;
  const json = JSON.stringify(State.nn.toJSON(), null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `nn_model_epoch${State.nn.epoch}.json`;
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
      State.activation = data.activation;

      networkViz.setNetwork(State.nn);
      boundaryViz.setNetwork(State.nn);
      buildLayerEditor();
      updateStats();
      updateWeightsDisplay();
      charts.update(State.nn);
      showToast(`📂 Model imported! Epoch: ${State.nn.epoch}`);
    } catch (err) {
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
