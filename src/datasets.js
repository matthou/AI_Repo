/**
 * Dataset Generators v2
 * Adds: Three-class spiral, Two Moons, 1D regression datasets.
 * All 2D datasets return { x, y, label } objects.
 * Regression datasets return { x, y } (y is the target value).
 */

const Datasets = {

  // ─── XOR ──────────────────────────────────────────────────────────────────
  xor: {
    label: 'XOR',
    classes: 2,
    generate(n = 300, noise = 0.1) {
      const data = [];
      for (let i = 0; i < n; i++) {
        const x = Math.random() * 2 - 1;
        const y = Math.random() * 2 - 1;
        const nx = x + randn() * noise;
        const ny = y + randn() * noise;
        data.push({ x: nx, y: ny, label: (x > 0) === (y > 0) ? 0 : 1 });
      }
      return data;
    }
  },

  // ─── 2-Class Spiral ───────────────────────────────────────────────────────
  spiral: {
    label: 'Spiral (2-class)',
    classes: 2,
    generate(n = 300, noise = 0.1) {
      const data = [];
      const half = Math.floor(n / 2);
      const addSpiral = (label, deltaT) => {
        for (let i = 0; i < half; i++) {
          const r = (i / half) * 0.95;
          const t = ((i / half) * 3.5 + deltaT) * Math.PI;
          data.push({
            x: r * Math.sin(t) + randn() * noise,
            y: r * Math.cos(t) + randn() * noise,
            label
          });
        }
      };
      addSpiral(0, 0);
      addSpiral(1, 1);
      return data;
    }
  },

  // ─── 3-Class Spiral ───────────────────────────────────────────────────────
  spiral3: {
    label: 'Spiral (3-class)',
    classes: 3,
    generate(n = 300, noise = 0.08) {
      const data = [];
      const perClass = Math.floor(n / 3);
      for (let c = 0; c < 3; c++) {
        for (let i = 0; i < perClass; i++) {
          const r = (i / perClass) * 0.92;
          const t = ((i / perClass) * 3.5 + c * (2 / 3)) * Math.PI * 2;
          data.push({
            x: r * Math.sin(t) + randn() * noise,
            y: r * Math.cos(t) + randn() * noise,
            label: c
          });
        }
      }
      return data;
    }
  },

  // ─── Circles ──────────────────────────────────────────────────────────────
  circles: {
    label: 'Circles',
    classes: 2,
    generate(n = 300, noise = 0.08) {
      const data = [];
      const half = Math.floor(n / 2);
      for (let i = 0; i < half; i++) {
        const angle = Math.random() * 2 * Math.PI;
        const r = Math.random() * 0.4;
        data.push({ x: r * Math.cos(angle) + randn() * noise, y: r * Math.sin(angle) + randn() * noise, label: 0 });
      }
      for (let i = 0; i < half; i++) {
        const angle = Math.random() * 2 * Math.PI;
        const r = 0.6 + Math.random() * 0.35;
        data.push({ x: r * Math.cos(angle) + randn() * noise, y: r * Math.sin(angle) + randn() * noise, label: 1 });
      }
      return data;
    }
  },

  // ─── Two Moons ────────────────────────────────────────────────────────────
  moons: {
    label: 'Two Moons',
    classes: 2,
    generate(n = 300, noise = 0.1) {
      const data = [];
      const half = Math.floor(n / 2);
      for (let i = 0; i < half; i++) {
        const t = Math.PI * i / half;
        data.push({
          x: Math.cos(t) * 0.8 + randn() * noise,
          y: Math.sin(t) * 0.8 - 0.2 + randn() * noise,
          label: 0
        });
      }
      for (let i = 0; i < half; i++) {
        const t = Math.PI * i / half;
        data.push({
          x: 1 - Math.cos(t) * 0.8 + randn() * noise - 0.5,
          y: -Math.sin(t) * 0.8 + 0.2 + randn() * noise,
          label: 1
        });
      }
      return data;
    }
  },

  // ─── Gaussian Blobs ───────────────────────────────────────────────────────
  gaussian: {
    label: 'Gaussian',
    classes: 2,
    generate(n = 300, noise = 0.15) {
      const data = [];
      const half = Math.floor(n / 2);
      for (let i = 0; i < half; i++) {
        data.push({ x: randn() * (noise + 0.2) - 0.4, y: randn() * (noise + 0.2), label: 0 });
      }
      for (let i = 0; i < half; i++) {
        data.push({ x: randn() * (noise + 0.2) + 0.4, y: randn() * (noise + 0.2), label: 1 });
      }
      return data;
    }
  },

  // ─── Checkerboard ─────────────────────────────────────────────────────────
  checkerboard: {
    label: 'Checkerboard',
    classes: 2,
    generate(n = 300, noise = 0.04) {
      const data = [];
      for (let i = 0; i < n; i++) {
        const x = Math.random() * 2 - 1;
        const y = Math.random() * 2 - 1;
        const cx = Math.floor((x + 1) * 2);
        const cy = Math.floor((y + 1) * 2);
        data.push({ x: x + randn() * noise, y: y + randn() * noise, label: (cx + cy) % 2 });
      }
      return data;
    }
  },

  // ─── Ring of Clusters ─────────────────────────────────────────────────────
  ringclusters: {
    label: 'Ring Clusters',
    classes: 2,
    generate(n = 300, noise = 0.06) {
      const data = [];
      const half = Math.floor(n / 2);
      const clusterCount = 4;
      for (let i = 0; i < half; i++) {
        const c = i % clusterCount;
        const a = (c / clusterCount) * Math.PI * 2;
        data.push({
          x: Math.cos(a) * 0.7 + randn() * noise,
          y: Math.sin(a) * 0.7 + randn() * noise,
          label: 0
        });
      }
      for (let i = 0; i < half; i++) {
        const c = i % clusterCount;
        const a = (c / clusterCount) * Math.PI * 2 + Math.PI / clusterCount;
        data.push({
          x: Math.cos(a) * 0.4 + randn() * noise,
          y: Math.sin(a) * 0.4 + randn() * noise,
          label: 1
        });
      }
      return data;
    }
  },

  // ─── Regression Datasets ──────────────────────────────────────────────────
  sin: {
    label: 'Sine Wave (Regression)',
    classes: 1,
    problemType: 'regression',
    generate(n = 300, noise = 0.1) {
      const data = [];
      for (let i = 0; i < n; i++) {
        const x = Math.random() * 2 - 1; // [-1, 1]
        const y = Math.sin(x * Math.PI) + randn() * noise;
        data.push({ x, y, label: 0 }); // label is dummy
      }
      return data;
    }
  },

  cubic: {
    label: 'Cubic Polynomial (Regression)',
    classes: 1,
    problemType: 'regression',
    generate(n = 300, noise = 0.1) {
      const data = [];
      for (let i = 0; i < n; i++) {
        const x = Math.random() * 2 - 1;
        const y = (x * x * x - 0.7 * x) * 2 + randn() * noise;
        data.push({ x, y, label: 0 });
      }
      return data;
    }
  },

  step: {
    label: 'Step Function (Regression)',
    classes: 1,
    problemType: 'regression',
    generate(n = 300, noise = 0.05) {
      const data = [];
      for (let i = 0; i < n; i++) {
        const x = Math.random() * 2 - 1;
        const y = (x >= 0 ? 0.6 : -0.6) + randn() * noise;
        data.push({ x, y, label: 0 });
      }
      return data;
    }
  }
};

// ─── Utility ──────────────────────────────────────────────────────────────────

function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Convert raw data points to {input, target, label, x, y} format.
 * Targets are one-hot encoded for multiclass.
 */
function prepareDataset(rawData, numClasses, problemType = 'classification') {
  const shuffled = rawData.slice().sort(() => Math.random() - 0.5);
  return shuffled.map(pt => {
    if (problemType === 'regression') {
      return { input: [pt.x], target: [pt.y], label: 0, x: pt.x, y: pt.y };
    }
    let target;
    if (numClasses > 2) {
      target = Array(numClasses).fill(0);
      target[pt.label] = 1;
    } else {
      target = [pt.label];
    }
    return { input: [pt.x, pt.y], target, label: pt.label, x: pt.x, y: pt.y };
  });
}

function trainTestSplit(data, trainRatio = 0.8) {
  const n = Math.floor(data.length * trainRatio);
  return { train: data.slice(0, n), test: data.slice(n) };
}
