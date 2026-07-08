/**
 * Neural Network Engine — v2
 * Adds: Dropout, L2 regularization, Softmax + categorical CE,
 *       LR scheduler (cosine annealing), multiclass support.
 */

// ─── Activation Functions ──────────────────────────────────────────────────

const Activations = {
  sigmoid: {
    fn: x => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))),
    deriv: x => { const s = 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))); return s * (1 - s); },
    label: 'Sigmoid'
  },
  relu: {
    fn: x => Math.max(0, x),
    deriv: x => x > 0 ? 1 : 0,
    label: 'ReLU'
  },
  tanh: {
    fn: x => Math.tanh(x),
    deriv: x => 1 - Math.tanh(x) ** 2,
    label: 'Tanh'
  },
  leakyrelu: {
    fn: x => x > 0 ? x : 0.01 * x,
    deriv: x => x > 0 ? 1 : 0.01,
    label: 'Leaky ReLU'
  },
  elu: {
    fn: x => x >= 0 ? x : Math.exp(x) - 1,
    deriv: x => x >= 0 ? 1 : Math.exp(x),
    label: 'ELU'
  },
  swish: {
    fn: x => x / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))),
    deriv: x => {
      const s = 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
      return s + x * s * (1 - s);
    },
    label: 'Swish'
  },
  linear: {
    fn: x => x,
    deriv: () => 1,
    label: 'Linear'
  }
};

// ─── Softmax ───────────────────────────────────────────────────────────────

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

// ─── Loss Functions ────────────────────────────────────────────────────────

function mseLoss(predictions, targets) {
  let loss = 0;
  for (let i = 0; i < predictions.length; i++) loss += (predictions[i] - targets[i]) ** 2;
  return loss / predictions.length;
}

function binaryCrossEntropy(predictions, targets) {
  let loss = 0;
  for (let i = 0; i < predictions.length; i++) {
    const p = Math.max(1e-7, Math.min(1 - 1e-7, predictions[i]));
    loss -= targets[i] * Math.log(p) + (1 - targets[i]) * Math.log(1 - p);
  }
  return loss / predictions.length;
}

function categoricalCrossEntropy(probs, targetOneHot) {
  let loss = 0;
  for (let i = 0; i < probs.length; i++) {
    loss -= targetOneHot[i] * Math.log(Math.max(1e-7, probs[i]));
  }
  return loss;
}

// ─── Weight Initialization ─────────────────────────────────────────────────

function glorotUniform(fanIn, fanOut) {
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  return () => (Math.random() * 2 - 1) * limit;
}

function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ─── Neural Network Class ──────────────────────────────────────────────────

class NeuralNetwork {
  /**
   * @param {number[]} layerSizes       - Array of neuron counts per layer (including input)
   * @param {string}   activation       - Hidden layer activation key
   * @param {string}   outputActivation - 'sigmoid' | 'softmax' | 'linear'
   * @param {Object}   options
   * @param {number}   options.dropout  - Dropout rate 0–1 (applied to hidden layers during training)
   * @param {number}   options.l2       - L2 weight decay coefficient
   */
  constructor(
    layerSizes = [2, 4, 4, 1],
    activation = 'tanh',
    outputActivation = 'sigmoid',
    { dropout = 0, l2 = 0 } = {}
  ) {
    this.layerSizes = layerSizes;
    this.activation = activation;
    this.outputActivation = outputActivation;
    this.dropout = dropout;
    this.l2 = l2;
    this.numLayers = layerSizes.length;

    this.weights = [];
    this.biases = [];
    this.zValues = [];
    this.aValues = [];
    this.dropoutMasks = [];
    this.gradWeights = [];
    this.gradBiases = [];

    // Adam state
    this.mWeights = [];
    this.vWeights = [];
    this.mBiases = [];
    this.vBiases = [];
    this.adamT = 0;

    // Metrics
    this.lossHistory = [];
    this.accHistory = [];
    this.epoch = 0;
    this.weightGradMagnitudes = [];

    // LR schedule
    this.baseLR = null;   // set by scheduler
    this.lrSchedule = 'constant';  // 'constant' | 'cosine' | 'step'
    this.totalEpochs = 500;

    this.initWeights();
  }

  initWeights() {
    this.weights = [];
    this.biases = [];
    this.mWeights = [];
    this.vWeights = [];
    this.mBiases = [];
    this.vBiases = [];
    this.adamT = 0;

    for (let l = 0; l < this.numLayers - 1; l++) {
      const fanIn = this.layerSizes[l];
      const fanOut = this.layerSizes[l + 1];
      const initFn = glorotUniform(fanIn, fanOut);

      const W = Array.from({ length: fanOut }, () =>
        Array.from({ length: fanIn }, initFn)
      );
      const b = Array.from({ length: fanOut }, () => 0);

      this.weights.push(W);
      this.biases.push(b);

      this.mWeights.push(Array.from({ length: fanOut }, () => Array(fanIn).fill(0)));
      this.vWeights.push(Array.from({ length: fanOut }, () => Array(fanIn).fill(0)));
      this.mBiases.push(Array(fanOut).fill(0));
      this.vBiases.push(Array(fanOut).fill(0));
    }

    this.lossHistory = [];
    this.accHistory = [];
    this.epoch = 0;
    this.weightGradMagnitudes = this.weights.map(W => W.map(row => row.map(() => 0)));
  }

  // ─── LR Schedule ────────────────────────────────────────────────────────

  _getCurrentLR(baseLR) {
    if (this.lrSchedule === 'constant') return baseLR;

    const t = this.epoch;
    const T = this.totalEpochs;

    if (this.lrSchedule === 'cosine') {
      return baseLR * 0.5 * (1 + Math.cos(Math.PI * t / T));
    }

    if (this.lrSchedule === 'step') {
      // Halve every 100 epochs
      const factor = Math.pow(0.5, Math.floor(t / 100));
      return baseLR * factor;
    }

    return baseLR;
  }

  // ─── Forward Pass ────────────────────────────────────────────────────────

  forward(input, training = false) {
    this.aValues = [input.slice()];
    this.zValues = [];
    this.dropoutMasks = [];

    for (let l = 0; l < this.numLayers - 1; l++) {
      const W = this.weights[l];
      const b = this.biases[l];
      const aIn = this.aValues[l];
      const isOutput = l === this.numLayers - 2;
      const isInput  = l === 0;

      // Pre-activation (z)
      const z = W.map((wRow, j) =>
        wRow.reduce((sum, w, k) => sum + w * aIn[k], 0) + b[j]
      );
      this.zValues.push(z);

      let a;
      if (isOutput && this.outputActivation === 'softmax') {
        a = softmax(z);
      } else {
        const actKey = isOutput ? this.outputActivation : this.activation;
        a = z.map(zj => Activations[actKey].fn(zj));
      }

      // Dropout on hidden layers during training
      if (training && !isOutput && this.dropout > 0) {
        const mask = a.map(() => Math.random() > this.dropout ? 1 : 0);
        this.dropoutMasks.push(mask);
        a = a.map((v, j) => v * mask[j] / (1 - this.dropout));
      } else {
        this.dropoutMasks.push(a.map(() => 1));
      }

      this.aValues.push(a);
    }

    return this.aValues[this.aValues.length - 1];
  }

  predict(input) {
    // Stateless prediction (no dropout, no stored state)
    let a = input.slice();
    for (let l = 0; l < this.numLayers - 1; l++) {
      const W = this.weights[l];
      const b = this.biases[l];
      const isOutput = l === this.numLayers - 2;

      const z = W.map((wRow, j) =>
        wRow.reduce((sum, w, k) => sum + w * a[k], 0) + b[j]
      );

      if (isOutput && this.outputActivation === 'softmax') {
        a = softmax(z);
      } else {
        const actKey = isOutput ? this.outputActivation : this.activation;
        a = z.map(zj => Activations[actKey].fn(zj));
      }
    }
    return a;
  }

  // ─── Backward Pass ───────────────────────────────────────────────────────

  backward(targets) {
    const L = this.numLayers - 1;
    const gW = this.weights.map(W => W.map(row => row.map(() => 0)));
    const gB = this.biases.map(b => b.map(() => 0));

    // Output delta
    let delta;
    if (this.outputActivation === 'softmax') {
      // d(CE)/d(softmax_input) = prob - target (simplified combined gradient)
      delta = this.aValues[L].map((a, j) => a - targets[j]);
    } else {
      delta = this.aValues[L].map((a, j) => {
        const dAct = Activations[this.outputActivation].deriv(this.zValues[L - 1][j]);
        return (a - targets[j]) * dAct;
      });
    }

    for (let l = L - 1; l >= 0; l--) {
      const aIn = this.aValues[l];

      for (let j = 0; j < delta.length; j++) {
        gB[l][j] += delta[j];
        for (let k = 0; k < aIn.length; k++) {
          gW[l][j][k] += delta[j] * aIn[k];
        }
      }

      if (l > 0) {
        const W = this.weights[l];
        const actKey = this.activation;
        const mask = this.dropoutMasks[l - 1] || aIn.map(() => 1);
        delta = aIn.map((_, k) => {
          const grad = W.reduce((sum, wRow, j) => sum + wRow[k] * delta[j], 0);
          const dAct = Activations[actKey].deriv(this.zValues[l - 1][k]);
          return grad * dAct * mask[k];
        });
      }
    }

    // L2 regularization gradients
    if (this.l2 > 0) {
      for (let l = 0; l < this.weights.length; l++) {
        for (let j = 0; j < this.weights[l].length; j++) {
          for (let k = 0; k < this.weights[l][j].length; k++) {
            gW[l][j][k] += this.l2 * this.weights[l][j][k];
          }
        }
      }
    }

    this.gradWeights = gW;
    this.gradBiases = gB;
    this.weightGradMagnitudes = gW.map(layer => layer.map(row => row.map(g => Math.abs(g))));
  }

  // ─── Optimizer Steps ──────────────────────────────────────────────────────

  _sgdUpdate(lr) {
    for (let l = 0; l < this.weights.length; l++) {
      for (let j = 0; j < this.weights[l].length; j++) {
        this.biases[l][j] -= lr * this.gradBiases[l][j];
        for (let k = 0; k < this.weights[l][j].length; k++) {
          this.weights[l][j][k] -= lr * this.gradWeights[l][j][k];
        }
      }
    }
  }

  _adamUpdate(lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.adamT++;
    const t = this.adamT;
    for (let l = 0; l < this.weights.length; l++) {
      for (let j = 0; j < this.weights[l].length; j++) {
        const gb = this.gradBiases[l][j];
        this.mBiases[l][j] = beta1 * this.mBiases[l][j] + (1 - beta1) * gb;
        this.vBiases[l][j] = beta2 * this.vBiases[l][j] + (1 - beta2) * gb * gb;
        const mHatB = this.mBiases[l][j] / (1 - beta1 ** t);
        const vHatB = this.vBiases[l][j] / (1 - beta2 ** t);
        this.biases[l][j] -= lr * mHatB / (Math.sqrt(vHatB) + eps);

        for (let k = 0; k < this.weights[l][j].length; k++) {
          const gw = this.gradWeights[l][j][k];
          this.mWeights[l][j][k] = beta1 * this.mWeights[l][j][k] + (1 - beta1) * gw;
          this.vWeights[l][j][k] = beta2 * this.vWeights[l][j][k] + (1 - beta2) * gw * gw;
          const mHat = this.mWeights[l][j][k] / (1 - beta1 ** t);
          const vHat = this.vWeights[l][j][k] / (1 - beta2 ** t);
          this.weights[l][j][k] -= lr * mHat / (Math.sqrt(vHat) + eps);
        }
      }
    }
  }

  // ─── Training Step ───────────────────────────────────────────────────────

  trainBatch(inputs, targets, learningRate = 0.01, optimizer = 'adam') {
    const effectiveLR = this._getCurrentLR(learningRate);

    const accGW = this.weights.map(W => W.map(row => row.map(() => 0)));
    const accGB = this.biases.map(b => b.map(() => 0));

    let totalLoss = 0;
    let correct = 0;
    
    // For R^2 score computation in regression
    const yTrueList = [];
    const yPredList = [];

    for (let i = 0; i < inputs.length; i++) {
      const pred = this.forward(inputs[i], true /* training=true */);
      const tgt = targets[i];

      if (this.outputActivation === 'softmax') {
        totalLoss += categoricalCrossEntropy(pred, tgt);
        const predClass = pred.indexOf(Math.max(...pred));
        const trueClass = tgt.indexOf(Math.max(...tgt));
        if (predClass === trueClass) correct++;
      } else if (this.outputActivation === 'linear') {
        totalLoss += mseLoss(pred, tgt);
        yTrueList.push(tgt[0]);
        yPredList.push(pred[0]);
      } else {
        totalLoss += binaryCrossEntropy(pred, tgt);
        if ((pred[0] > 0.5 ? 1 : 0) === tgt[0]) correct++;
      }

      this.backward(tgt);

      for (let l = 0; l < accGW.length; l++) {
        for (let j = 0; j < accGW[l].length; j++) {
          accGB[l][j] += this.gradBiases[l][j];
          for (let k = 0; k < accGW[l][j].length; k++) {
            accGW[l][j][k] += this.gradWeights[l][j][k];
          }
        }
      }
    }

    const n = inputs.length;
    this.gradWeights = accGW.map(l => l.map(row => row.map(g => g / n)));
    this.gradBiases = accGB.map(l => l.map(g => g / n));
    this.weightGradMagnitudes = this.gradWeights.map(layer =>
      layer.map(row => row.map(g => Math.abs(g)))
    );

    if (optimizer === 'adam') {
      this._adamUpdate(effectiveLR);
    } else {
      this._sgdUpdate(effectiveLR);
    }

    const loss = totalLoss / n;
    
    let acc;
    if (this.outputActivation === 'linear') {
      const yMean = yTrueList.reduce((a, b) => a + b, 0) / n;
      const ssTot = yTrueList.reduce((sum, val) => sum + (val - yMean) ** 2, 0);
      const ssRes = yTrueList.reduce((sum, val, idx) => sum + (val - yPredList[idx]) ** 2, 0);
      acc = ssTot > 1e-8 ? 1 - (ssRes / ssTot) : 0;
    } else {
      acc = correct / n;
    }

    this.lossHistory.push(loss);
    this.accHistory.push(acc);
    this.epoch++;

    return { loss, acc, effectiveLR };
  }

  // ─── Evaluate on Test Set ────────────────────────────────────────────────

  evaluate(inputs, targets) {
    let totalLoss = 0;
    let correct = 0;
    const numClasses = targets[0].length;
    
    let confusionMatrix = null;
    if (this.outputActivation !== 'linear') {
      confusionMatrix = Array.from({ length: numClasses }, () => Array(numClasses).fill(0));
    }

    const yTrueList = [];
    const yPredList = [];

    for (let i = 0; i < inputs.length; i++) {
      const pred = this.predict(inputs[i]);
      const tgt = targets[i];

      if (this.outputActivation === 'softmax') {
        totalLoss += categoricalCrossEntropy(pred, tgt);
        const predClass = pred.indexOf(Math.max(...pred));
        const trueClass = tgt.indexOf(Math.max(...tgt));
        if (predClass === trueClass) correct++;
        confusionMatrix[trueClass][predClass]++;
      } else if (this.outputActivation === 'linear') {
        totalLoss += mseLoss(pred, tgt);
        yTrueList.push(tgt[0]);
        yPredList.push(pred[0]);
      } else {
        totalLoss += binaryCrossEntropy(pred, tgt);
        const predClass = pred[0] > 0.5 ? 1 : 0;
        const trueClass = tgt[0];
        if (predClass === trueClass) correct++;
        confusionMatrix[trueClass][predClass]++;
      }
    }

    let acc;
    if (this.outputActivation === 'linear') {
      const n = inputs.length;
      const yMean = yTrueList.reduce((a, b) => a + b, 0) / n;
      const ssTot = yTrueList.reduce((sum, val) => sum + (val - yMean) ** 2, 0);
      const ssRes = yTrueList.reduce((sum, val, idx) => sum + (val - yPredList[idx]) ** 2, 0);
      acc = ssTot > 1e-8 ? 1 - (ssRes / ssTot) : 0;
    } else {
      acc = correct / inputs.length;
    }

    return {
      loss: totalLoss / inputs.length,
      acc,
      confusionMatrix
    };
  }

  // ─── Serialization ───────────────────────────────────────────────────────

  toJSON() {
    return {
      layerSizes: this.layerSizes,
      activation: this.activation,
      outputActivation: this.outputActivation,
      dropout: this.dropout,
      l2: this.l2,
      weights: this.weights,
      biases: this.biases,
      lossHistory: this.lossHistory,
      accHistory: this.accHistory,
      epoch: this.epoch,
      lrSchedule: this.lrSchedule
    };
  }

  static fromJSON(data) {
    const nn = new NeuralNetwork(
      data.layerSizes, data.activation, data.outputActivation,
      { dropout: data.dropout || 0, l2: data.l2 || 0 }
    );
    nn.weights = data.weights;
    nn.biases = data.biases;
    nn.lossHistory = data.lossHistory;
    nn.accHistory = data.accHistory;
    nn.epoch = data.epoch;
    nn.lrSchedule = data.lrSchedule || 'constant';
    return nn;
  }

  getActivations(input) {
    this.forward(input, false);
    return this.aValues;
  }

  // ─── Param count ────────────────────────────────────────────────────────

  paramCount() {
    let p = 0;
    for (let l = 0; l < this.weights.length; l++) {
      p += this.weights[l].length * this.weights[l][0].length + this.biases[l].length;
    }
    return p;
  }
}
