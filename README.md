# Neural-Network-Hybrid-Optimizer
A PyTorch implementation of a novel Hybrid Optimizer combining RMSProp with Particle Swarm Optimization (PSO). Achieves Adam-level performance on CIFAR-10 through dynamic swarm annealing and momentum integration.
# 🧠 Hybrid-Swarm-Optimizer
### Bridging Gradient Descent and Bio-Inspired Swarm Intelligence

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](./LICENSE)

## 📖 Overview
This project explores the intersection of mathematical calculus and biological swarm intelligence. We propose and implement a **Hybrid RMSProp-PSO Optimizer** that integrates the exploration capabilities of Particle Swarm Optimization (PSO) directly into the gradient descent update step of RMSProp.

By introducing **Dynamic Swarm Decay** and **Nesterov Momentum**, our hybrid architecture achieves convergence rates and accuracy comparable to industry-standard optimizers (AdamW) on the CIFAR-10 dataset, while offering greater robustness against local minima.

## 🚀 Key Features
* **Novel Architecture:** A custom PyTorch Optimizer class (`RMSProp_Hybrid`) fusing gradient vectors with swarm velocity vectors.
* **Dynamic Annealing:** Implements a decay factor $\frac{1}{1 + k \cdot t}$ to transition from high exploration (swarm) to high precision (gradient) over time.
* **Cognitive Memory:** Each parameter tracks its own "Personal Best" (pBest) value throughout training to guide weight updates.
* **Benchmarked:** Rigorously tested against SGD, RMSProp, and AdamW on the ResNet-18 and CNN architectures.

---

## 🔬 Methodology
The project was executed in three distinct research stages:

### Stage 1: Baseline Establishment 📊
We benchmarked standard optimizers (SGD, RMSProp, AdamW) to establish a performance floor and ceiling.
* **Result:** AdamW established the ceiling at ~77.74% accuracy.

### Stage 2: Meta-Tuning (Swarm Intelligence) 🐝
We employed **PSO**, **Grey Wolf Optimizer (GWO)**, and **Bee Colony Optimization (BCO)** to strictly tune the hyperparameters (LR, Alpha) of a standard RMSProp optimizer.
* **Finding:** Tuning improved RMSProp by **1.4%**, but revealed the "Horizon Effect" where high learning rates failed in longer training runs.

### Stage 3: Hybrid Architecting (The Solution) ⚡
We developed the `RMSProp_Hybrid` optimizer.
$$W_{new} = W_{old} - \eta \cdot \nabla_{RMS} + \phi \cdot (pBest - W_{old})$$
Where $\phi$ represents the swarm influence, which decays over time to prevent volatility in late-stage training.

---

## 📉 Results

| Optimizer | Configuration | Accuracy (CIFAR-10) | Notes |
| :--- | :--- | :--- | :--- |
| **Adam (Baseline)** | `lr=0.001` | **77.74%** | Industry Standard |
| **RMSProp (Default)** | `lr=0.001` | 72.95% | Prone to stagnation |
| **RMSProp (Tuned)** | `lr=0.0012` | 74.36% | Tuned via PSO (Stage 2) |
| **Hybrid (Ours)** | `c1=0.1, m=0.95` | **77.21%** | **Matches Adam performance** |

<p align="center">
  <img src="path/to/your/final_graph.png" alt="Training Comparison Graph" width="700">
  <br>
  <em>Figure 1: Convergence comparison between Adam (Blue) and Hybrid RMSProp-PSO (Orange) over 20 epochs.</em>
</p>

---

## 🛠️ Usage
