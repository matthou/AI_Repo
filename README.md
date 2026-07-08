# 🧠 Neural Network Playground

An interactive, browser-based neural network visualizer and trainer. Design custom architectures, watch real-time training animations, and understand deep learning intuitively.

## ✨ Features

- **Visual Network Designer** — Add/remove layers and neurons with a point-and-click interface
- **Live Training Animation** — Watch weights and activations update in real time as the network trains
- **Decision Boundary Visualization** — See the network's learned decision boundary animate on a 2D canvas
- **Multiple Datasets** — XOR, spiral, circles, and gaussian blobs
- **Backpropagation Visualizer** — Color-coded gradient flow through the network
- **Hyperparameter Controls** — Adjust learning rate, activation functions, batch size, and more
- **Loss / Accuracy Charts** — Real-time training metrics plotted live

## 🚀 Getting Started

Just open `index.html` in a browser — no build step, no server required!

```bash
git clone https://github.com/matthou/AI_Repo.git
cd AI_Repo
# Open index.html in your browser
start index.html   # Windows
open index.html    # macOS
```

## 🛠️ Tech Stack

- Vanilla HTML / CSS / JavaScript (zero dependencies!)
- Custom neural network engine implemented from scratch
- Canvas 2D API for real-time visualization

## 📁 Project Structure

```
AI_Repo/
├── index.html          # Main entry point
├── style.css           # Styling & animations
├── src/
│   ├── nn.js           # Neural network engine (forward/backward pass)
│   ├── datasets.js     # Dataset generators
│   ├── visualizer.js   # Network topology renderer
│   ├── boundary.js     # Decision boundary canvas
│   └── charts.js       # Loss/accuracy chart renderer
└── README.md
```

## 🎓 Learning Objectives

This playground is designed to help developers and students intuitively understand:
- How forward propagation works
- What backpropagation actually does
- How activation functions affect learning
- Why learning rate matters
- How decision boundaries form

## 📄 License

MIT
