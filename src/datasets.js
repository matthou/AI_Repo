/**
 * Dataset Generators
 * Generates 2D classification datasets for training the neural network.
 */

const Datasets = {

  // ─── XOR ──────────────────────────────────────────────────────────────────
  xor: {
    label: 'XOR',
    generate(n = 200, noise = 0.1) {
      const data = [];
      for (let i = 0; i < n; i++) {
        const x = (Math.random() * 2 - 1);
        const y = (Math.random() * 2 - 1);
        const nx = x + randn() * noise;
        const ny = y + randn() * noise;
        const label = (x > 0) === (y > 0) ? 0 : 1;
        data.push({ x: nx, y: ny, label });
      }
      return data;
    }
  },

  // ─── Spiral ───────────────────────────────────────────────────────────────
  spiral: {
    label: 'Spiral',
    generate(n = 200, noise = 0.1) {
      const data = [];
      const half = Math.floor(n / 2);

      const addSpiral = (label, deltaT) => {
        for (let i = 0; i < half; i++) {
          const r = (i / half) * 0.95;
          const t = ((i / half) * 3.5 + deltaT) * Math.PI;
          const x = r * Math.sin(t) + randn() * noise;
          const y = r * Math.cos(t) + randn() * noise;
          data.push({ x, y, label });
        }
      };

      addSpiral(0, 0);
      addSpiral(1, 1);
      return data;
    }
  },

  // ─── Circles ──────────────────────────────────────────────────────────────
  circles: {
    label: 'Circles',
    generate(n = 200, noise = 0.1) {
      const data = [];
      const half = Math.floor(n / 2);

      for (let i = 0; i < half; i++) {
        const angle = Math.random() * 2 * Math.PI;
        const r = Math.random() * 0.4;
        data.push({
          x: r * Math.cos(angle) + randn() * noise,
          y: r * Math.sin(angle) + randn() * noise,
          label: 0
        });
      }

      for (let i = 0; i < half; i++) {
        const angle = Math.random() * 2 * Math.PI;
        const r = 0.6 + Math.random() * 0.35;
        data.push({
          x: r * Math.cos(angle) + randn() * noise,
          y: r * Math.sin(angle) + randn() * noise,
          label: 1
        });
      }

      return data;
    }
  },

  // ─── Gaussians ────────────────────────────────────────────────────────────
  gaussian: {
    label: 'Gaussian',
    generate(n = 200, noise = 0.15) {
      const data = [];
      const half = Math.floor(n / 2);

      for (let i = 0; i < half; i++) {
        data.push({
          x: randn() * (noise + 0.2) - 0.4,
          y: randn() * (noise + 0.2),
          label: 0
        });
      }

      for (let i = 0; i < half; i++) {
        data.push({
          x: randn() * (noise + 0.2) + 0.4,
          y: randn() * (noise + 0.2),
          label: 1
        });
      }

      return data;
    }
  },

  // ─── Checkerboard ─────────────────────────────────────────────────────────
  checkerboard: {
    label: 'Checkerboard',
    generate(n = 200, noise = 0.05) {
      const data = [];
      for (let i = 0; i < n; i++) {
        const x = Math.random() * 2 - 1;
        const y = Math.random() * 2 - 1;
        const nx = x + randn() * noise;
        const ny = y + randn() * noise;
        const cx = Math.floor((x + 1) * 2);
        const cy = Math.floor((y + 1) * 2);
        const label = (cx + cy) % 2;
        data.push({ x: nx, y: ny, label });
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
 * Normalize dataset to [-1, 1] range and shuffle
 */
function prepareDataset(rawData) {
  // Shuffle
  const shuffled = rawData.slice().sort(() => Math.random() - 0.5);

  return shuffled.map(pt => ({
    input: [pt.x, pt.y],
    target: [pt.label],
    label: pt.label,
    x: pt.x,
    y: pt.y
  }));
}

/**
 * Split into train/test
 */
function trainTestSplit(data, trainRatio = 0.8) {
  const n = Math.floor(data.length * trainRatio);
  return {
    train: data.slice(0, n),
    test: data.slice(n)
  };
}
