/**
 * Neural Network Engine
 * Implements a fully-connected feedforward neural network
 * with forward pass, backpropagation, and multiple activation functions.
 */

// ─── Activation Functions ──────────────────────────────────────────────────

const Activations = {
  sigmoid: {
    fn: x => 1 / (1 + Math.exp(-x)),
    deriv: x => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); },
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
  linear: {
    fn: x => x,
    deriv: () => 1,
    label: 'Linear'
  }
};

// ─── Loss Functions ────────────────────────────────────────────────────────

function mseLoss(predictions, targets) {
  let loss = 0;
  for (let i = 0; i < predictions.length; i++) {
    loss += (predictions[i] - targets[i]) ** 2;
  }
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

// ─── Weight Initialization ─────────────────────────────────────────────────

function glorotUniform(fanIn, fanOut) {
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  return () => (Math.random() * 2 - 1) * limit;
}

function heNormal(fanIn) {
  const std = Math.sqrt(2 / fanIn);
  return () => randn() * std;
}

function randn() {
  // Box-Muller transform
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ─── Neural Network Class ──────────────────────────────────────────────────

class NeuralNetwork {
  /**
   * @param {number[]} layerSizes - Array of neuron counts per layer (including input)
   * @param {string} activation - Hidden layer activation function key
   * @param {string} outputActivation - Output layer activation function key
   */
  constructor(layerSizes = [2, 4, 4, 1], activation = 'tanh', outputActivation = 'sigmoid') {
    this.layerSizes = layerSizes;
    this.activation = activation;
    this.outputActivation = outputActivation;
    this.numLayers = layerSizes.length;

    this.weights = [];   // weights[l] is matrix [layerSizes[l+1] x layerSizes[l]]
    this.biases = [];    // biases[l] is vector [layerSizes[l+1]]

    // Training state
    this.zValues = [];   // pre-activations per layer
    this.aValues = [];   // post-activations per layer
    this.gradWeights = [];
    this.gradBiases = [];

    // Optimizer state (Adam)
    this.mWeights = [];
    this.vWeights = [];
    this.mBiases = [];
    this.vBiases = [];
    this.adamT = 0;

    // Training metrics
    this.lossHistory = [];
    this.accHistory = [];
    this.epoch = 0;

    // For visualization: gradient magnitudes per weight
    this.weightGradMagnitudes = [];

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

      // Weight matrix: rows = fanOut neurons, cols = fanIn neurons
      const W = Array.from({ length: fanOut }, () =>
        Array.from({ length: fanIn }, initFn)
      );
      const b = Array.from({ length: fanOut }, () => 0);

      this.weights.push(W);
      this.biases.push(b);

      // Adam moment vectors
      this.mWeights.push(Array.from({ length: fanOut }, () => Array(fanIn).fill(0)));
      this.vWeights.push(Array.from({ length: fanOut }, () => Array(fanIn).fill(0)));
      this.mBiases.push(Array(fanOut).fill(0));
      this.vBiases.push(Array(fanOut).fill(0));
    }

    this.lossHistory = [];
    this.accHistory = [];
    this.epoch = 0;
    this.weightGradMagnitudes = this.weights.map(W =>
      W.map(row => row.map(() => 0))
    );
  }

  // ─── Forward Pass ────────────────────────────────────────────────────────

  forward(input) {
    this.aValues = [input.slice()];
    this.zValues = [];

    for (let l = 0; l < this.numLayers - 1; l++) {
      const W = this.weights[l];
      const b = this.biases[l];
      const aIn = this.aValues[l];
      const isOutput = l === this.numLayers - 2;
      const actKey = isOutput ? this.outputActivation : this.activation;
      const act = Activations[actKey];

      const z = W.map((wRow, j) =>
        wRow.reduce((sum, w, k) => sum + w * aIn[k], 0) + b[j]
      );
      const a = z.map(zj => act.fn(zj));

      this.zValues.push(z);
      this.aValues.push(a);
    }

    return this.aValues[this.aValues.length - 1];
  }

  predict(input) {
    // Forward without storing state (for decision boundary queries)
    let a = input.slice();
    for (let l = 0; l < this.numLayers - 1; l++) {
      const W = this.weights[l];
      const b = this.biases[l];
      const isOutput = l === this.numLayers - 2;
      const actKey = isOutput ? this.outputActivation : this.activation;
      const act = Activations[actKey];
      a = W.map((wRow, j) =>
        act.fn(wRow.reduce((sum, w, k) => sum + w * a[k], 0) + b[j])
      );
    }
    return a;
  }

  // ─── Backward Pass ───────────────────────────────────────────────────────

  backward(targets, learningRate = 0.01, optimizer = 'adam') {
    const L = this.numLayers - 1; // number of weight layers
    const gW = this.weights.map(W => W.map(row => row.map(() => 0)));
    const gB = this.biases.map(b => b.map(() => 0));

    // Output layer delta
    let delta = this.aValues[L].map((a, j) => {
      const isOutput = true;
      const actKey = this.outputActivation;
      const dAct = Activations[actKey].deriv(this.zValues[L - 1][j]);
      return (a - targets[j]) * dAct;
    });

    // Backpropagate
    for (let l = L - 1; l >= 0; l--) {
      const aIn = this.aValues[l];

      // Compute gradients for this layer
      for (let j = 0; j < delta.length; j++) {
        gB[l][j] += delta[j];
        for (let k = 0; k < aIn.length; k++) {
          gW[l][j][k] += delta[j] * aIn[k];
        }
      }

      if (l > 0) {
        // Propagate delta backwards
        const W = this.weights[l];
        const actKey = this.activation;
        delta = aIn.map((_, k) => {
          const grad = W.reduce((sum, wRow, j) => sum + wRow[k] * delta[j], 0);
          return grad * Activations[actKey].deriv(this.zValues[l - 1][k]);
        });
      }
    }

    this.gradWeights = gW;
    this.gradBiases = gB;

    // Store gradient magnitudes for visualization
    this.weightGradMagnitudes = gW.map(layer =>
      layer.map(row => row.map(g => Math.abs(g)))
    );

    // Update weights
    if (optimizer === 'adam') {
      this._adamUpdate(learningRate);
    } else {
      this._sgdUpdate(learningRate);
    }
  }

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
        // Bias
        const gb = this.gradBiases[l][j];
        this.mBiases[l][j] = beta1 * this.mBiases[l][j] + (1 - beta1) * gb;
        this.vBiases[l][j] = beta2 * this.vBiases[l][j] + (1 - beta2) * gb * gb;
        const mHatB = this.mBiases[l][j] / (1 - beta1 ** t);
        const vHatB = this.vBiases[l][j] / (1 - beta2 ** t);
        this.biases[l][j] -= lr * mHatB / (Math.sqrt(vHatB) + eps);

        // Weights
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
    let totalLoss = 0;
    let correct = 0;

    // Reset gradients (accumulate over batch)
    const accGW = this.weights.map(W => W.map(row => row.map(() => 0)));
    const accGB = this.biases.map(b => b.map(() => 0));

    for (let i = 0; i < inputs.length; i++) {
      const pred = this.forward(inputs[i]);
      const tgt = targets[i];

      // Loss
      totalLoss += binaryCrossEntropy(pred, tgt);

      // Accuracy
      const predClass = pred[0] > 0.5 ? 1 : 0;
      if (predClass === tgt[0]) correct++;

      // Backward
      this.backward(tgt, learningRate, optimizer);

      // Accumulate gradients
      for (let l = 0; l < accGW.length; l++) {
        for (let j = 0; j < accGW[l].length; j++) {
          accGB[l][j] += this.gradBiases[l][j];
          for (let k = 0; k < accGW[l][j].length; k++) {
            accGW[l][j][k] += this.gradWeights[l][j][k];
          }
        }
      }
    }

    // Average gradients and apply
    const n = inputs.length;
    this.gradWeights = accGW.map(l => l.map(row => row.map(g => g / n)));
    this.gradBiases = accGB.map(l => l.map(g => g / n));
    this.weightGradMagnitudes = this.gradWeights.map(layer =>
      layer.map(row => row.map(g => Math.abs(g)))
    );

    if (optimizer === 'adam') {
      this._adamUpdate(learningRate);
    } else {
      this._sgdUpdate(learningRate);
    }

    const loss = totalLoss / inputs.length;
    const acc = correct / inputs.length;
    this.lossHistory.push(loss);
    this.accHistory.push(acc);
    this.epoch++;

    return { loss, acc };
  }

  // ─── Serialization ───────────────────────────────────────────────────────

  toJSON() {
    return {
      layerSizes: this.layerSizes,
      activation: this.activation,
      outputActivation: this.outputActivation,
      weights: this.weights,
      biases: this.biases,
      lossHistory: this.lossHistory,
      accHistory: this.accHistory,
      epoch: this.epoch
    };
  }

  static fromJSON(data) {
    const nn = new NeuralNetwork(data.layerSizes, data.activation, data.outputActivation);
    nn.weights = data.weights;
    nn.biases = data.biases;
    nn.lossHistory = data.lossHistory;
    nn.accHistory = data.accHistory;
    nn.epoch = data.epoch;
    return nn;
  }

  // ─── Get activation values for visualization ──────────────────────────────

  getActivations(input) {
    this.forward(input);
    return this.aValues;
  }
}
