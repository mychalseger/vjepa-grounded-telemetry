# Grounding Spatial Representations via Visual World Models for Robust Autoregressive Drift Reduction in Low-Power Robot Telemetry

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/VJEPA_Grounded_Spatial_Representation_Journal.pdf)

This repository contains the complete PyTorch implementation, experimental pipeline, training logs, evaluation suite, and full academic publication for **V-JEPA Grounded Spatial Representations**. 

The framework eliminates **autoregressive drift** in closed-loop robotic state forecasting by regularizing a lightweight, 2.46-million parameter student network against Meta's pre-trained **V-JEPA (Video Joint Embedding Predictive Architecture)** world model during offline training. At runtime, the model operates in complete **Visual Silence** (zero camera feeds and zero vision transformers at inference), achieving a **73.80% average reduction in trajectory drift** in closed-loop MuJoCo physics simulation.

---

## Key Results & Highlights

- **73.80% Spatial Drift Reduction:** Average L2 trajectory tracking error reduced from **147.55** (Baseline LSTM) to **38.66** across 21 held-out test episodes in closed-loop MuJoCo physics ($p < 10^{-6}$).
- **Peak Precision on Non-Linear Inflections:** Up to **87.27% error reduction** on sharp directional shifts (Episode 203).
- **Extreme Edge Efficiency:** Total deployed inference network comprises **2,455,810 trainable parameters** requiring only **9.82 MB of FP32 RAM** ($<$5 MB in FP16, $<$2.5 MB in INT8).
- **$124\times$ Parameter Compression:** $124\times$ smaller than the visual teacher model (V-JEPA ViT-L/16, ~304M parameters).
- **Runtime Visual Silence:** Complete immunity to visual occlusions, lighting shifts, lens smudges, and network latency at inference time.

---

## System Architecture

```
                                [ Training Phase Only ]
                  ┌─────────────────────────────────────────────────┐
                  │ 64-Frame Synchronized Video Clips (100Hz)       │
                  └────────────────────────┬────────────────────────┘
                                           │
                                           ▼ (Frozen V-JEPA ViT-L/16)
                  ┌─────────────────────────────────────────────────┐
                  │ 1024-Dim Visual Latent Anchor Vector (v_t)      │
                  └────────────────────────┬────────────────────────┘
                                           │
                                           │  Symmetric InfoNCE Loss (tau=0.07)
                                           │  + Scheduled Manifold Sampling (SMS)
                                           │
                  ┌────────────────────────┴────────────────────────┐
                  │ Projected Student Embedding (z_t) [1024-Dim]    │
                  └────────────────────────▲────────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        │ Student Residual MLP (802 -> 1024)   │
                        └──────────────────▲──────────────────┘
                                           │
                  ┌────────────────────────┴────────────────────────┐
                  │ Telemetry Window (W=40) + Scaled PE (0.1 x PE)  │
                  │ + Action Command Vector (a_t) [802-Dim Input]   │
                  └─────────────────────────────────────────────────┘
                                           │
                                [ Inference Phase ]
                                           │ (Visual Silence: v_t = None)
                                           ▼
                  ┌─────────────────────────────────────────────────┐
                  │ Attentive Sequence Decoder (Self-Attn + LSTM)   │
                  └────────────────────────┬────────────────────────┘
                                           │
                                           ▼
                  ┌─────────────────────────────────────────────────┐
                  │ 5-Step Continuous Future Trajectory Prediction  │
                  └────────────────────────┬────────────────────────┘
                                           │
                                           ▼
                  ┌─────────────────────────────────────────────────┐
                  │ Closed-Loop MuJoCo Physics Simulator Rollout    │
                  └─────────────────────────────────────────────────┘
```

---

## Repository Structure

```tree
repo/
├── README.md                           # This repository documentation
├── baseline_model.py                   # Ungrounded 2-layer Seq2Seq LSTM baseline forecaster
├── optimized_model.py                  # ContrastiveStudentMLP, ContrastiveDecoder & InfoNCE loss
├── data_pipeline.py                    # LeRobot dataset loader, V-JEPA feature extractor & normalizer
├── main.py                             # Master experiment runner: training, evaluation & logging
├── feature_cache.h5                    # Fast HDF5 cache with pre-extracted V-JEPA latents & normalized actions
├── logs.txt                            # Full execution, training convergence & evaluation logs
├── episode_evaluation_results.json     # Exact quantitative metrics across all 21 test episodes
├── paper/                              # Academic journal publication package
│   ├── VJEPA_Grounded_Spatial_Representation_Journal.pdf  # 17-page compiled journal paper PDF
│   ├── VJEPA_Grounded_Spatial_Representation_Journal.tex  # LaTeX source document
│   └── VJEPA_Grounded_Spatial_Representation_Journal.md  # Markdown publication companion
└── results/                            # Evaluation trajectory plots & animated rollout GIFs
    ├── trajectory_comparison_ep_185.png to ep_205.png     # 21 static side-by-side trajectory comparisons
    └── trajectory_animation_ep_185.gif to ep_205.gif       # 21 animated closed-loop MuJoCo rollouts
```

---

## Installation & Environment Setup

### 1. Prerequisites
- Python 3.10 or higher
- PyTorch 2.0+ (CUDA GPU acceleration recommended)
- MuJoCo 3.0+

### 2. Install Dependencies
```bash
pip install torch torchvision numpy h5py matplotlib tqdm lerobot mujoco
```

---

## Running the Experiments

### Master Pipeline Execution
To execute the complete end-to-end experiment (cache verification, baseline training, V-JEPA grounded model training, and 21-episode closed-loop MuJoCo rollout):

```bash
python main.py
```

### Module Breakdown
1. **Feature Cache Generation (`data_pipeline.py`):**  
   Extracts 1024-dimensional visual representations from 64-frame video clips using Meta's `facebook/vjepa2-vitl-fpc64-256` model, standardizes joint positions, velocities, and actions exclusively over training episodes (0–163), and caches to `feature_cache.h5`.
2. **Baseline Model Training (`baseline_model.py`):**  
   Trains an ungrounded 2-layer Sequence-to-Sequence LSTM for 200 epochs using Mean Squared Error loss.
3. **Grounded Model Training (`optimized_model.py`):**  
   Trains the dual-tower student model for 120 epochs using the joint multi-task loss `L_total = L_MSE + 0.25 * L_InfoNCE`, with Scheduled Manifold Sampling decay (`λ_blend → 0.500`) and stochastic trajectory jitter (`σ = 0.02`).
4. **Closed-Loop MuJoCo Evaluation (`main.py`):**  
   Evaluates both models under strict **Visual Silence** across 21 held-out test episodes (Episodes 185 to 205), generating comparison plots and animations in `results/`.

---

## Quantitative Benchmark Evaluation

Closed-loop tracking drift (average L2 Euclidean error between ground-truth trajectory and simulator rollout) across all **21 held-out test episodes**:

| Test Episode ID | Baseline LSTM Drift | V-JEPA Grounded Drift | Drift Reduction (%) | Trajectory Plot |
| :---: | :---: | :---: | :---: | :---: |
| **Episode 185** | 93.41 | **24.68** | **73.58%** | [`results/trajectory_comparison_ep_185.png`](results/trajectory_comparison_ep_185.png) |
| **Episode 186** | 187.12 | **42.05** | **77.53%** | [`results/trajectory_comparison_ep_186.png`](results/trajectory_comparison_ep_186.png) |
| **Episode 187** | 142.69 | **37.79** | **73.52%** | [`results/trajectory_comparison_ep_187.png`](results/trajectory_comparison_ep_187.png) |
| **Episode 188** | 147.56 | **51.47** | **65.12%** | [`results/trajectory_comparison_ep_188.png`](results/trajectory_comparison_ep_188.png) |
| **Episode 189** | 85.18 | **40.23** | **52.77%** | [`results/trajectory_comparison_ep_189.png`](results/trajectory_comparison_ep_189.png) |
| **Episode 190** | 216.52 | **36.01** | **83.37%** | [`results/trajectory_comparison_ep_190.png`](results/trajectory_comparison_ep_190.png) |
| **Episode 191** | 150.67 | **47.72** | **68.33%** | [`results/trajectory_comparison_ep_191.png`](results/trajectory_comparison_ep_191.png) |
| **Episode 192** | 177.28 | **48.88** | **72.43%** | [`results/trajectory_comparison_ep_192.png`](results/trajectory_comparison_ep_192.png) |
| **Episode 193** | 134.78 | **38.55** | **71.40%** | [`results/trajectory_comparison_ep_193.png`](results/trajectory_comparison_ep_193.png) |
| **Episode 194** | 154.33 | **39.08** | **74.68%** | [`results/trajectory_comparison_ep_194.png`](results/trajectory_comparison_ep_194.png) |
| **Episode 195** | 166.13 | **45.27** | **72.75%** | [`results/trajectory_comparison_ep_195.png`](results/trajectory_comparison_ep_195.png) |
| **Episode 196** | 137.67 | **22.94** | **83.34%** | [`results/trajectory_comparison_ep_196.png`](results/trajectory_comparison_ep_196.png) |
| **Episode 197** | 62.38 | **34.83** | **44.17%** | [`results/trajectory_comparison_ep_197.png`](results/trajectory_comparison_ep_197.png) |
| **Episode 198** | 128.91 | **34.31** | **73.39%** | [`results/trajectory_comparison_ep_198.png`](results/trajectory_comparison_ep_198.png) |
| **Episode 199** | 157.43 | **53.08** | **66.28%** | [`results/trajectory_comparison_ep_199.png`](results/trajectory_comparison_ep_199.png) |
| **Episode 200** | 186.98 | **41.69** | **77.70%** | [`results/trajectory_comparison_ep_200.png`](results/trajectory_comparison_ep_200.png) |
| **Episode 201** | 127.87 | **25.12** | **80.36%** | [`results/trajectory_comparison_ep_201.png`](results/trajectory_comparison_ep_201.png) |
| **Episode 202** | 143.18 | **44.51** | **68.91%** | [`results/trajectory_comparison_ep_202.png`](results/trajectory_comparison_ep_202.png) |
| **Episode 203** | 206.56 | **26.30** | **87.27%** | [`results/trajectory_comparison_ep_203.png`](results/trajectory_comparison_ep_203.png) |
| **Episode 204** | 215.29 | **40.05** | **81.40%** | [`results/trajectory_comparison_ep_204.png`](results/trajectory_comparison_ep_204.png) |
| **Episode 205** | 76.73 | **37.30** | **51.39%** | [`results/trajectory_comparison_ep_205.png`](results/trajectory_comparison_ep_205.png) |
| **OVERALL AVERAGE** | **147.55 $\pm$ 43.8** | **38.66 $\pm$ 8.6** | **73.80%** | — |

---

## Computational Complexity & Parameter Allocation

| Module Subsystem | Layer Architecture | Trainable Parameters | Proportion |
| :--- | :--- | :---: | :---: |
| **Student MLP Input** | Linear ($802 \to 512$) + LayerNorm + GELU | 412,160 | 16.78% |
| **Residual Block 1** | Two Linear ($512 \to 512$) + LayerNorm + GELU | 527,360 | 21.47% |
| **Residual Block 2** | Two Linear ($512 \to 512$) + LayerNorm + GELU | 527,360 | 21.47% |
| **Representation Head** | Linear ($512 \to 1024$) + LayerNorm | 527,360 | 21.47% |
| **Total Student Network** | `ContrastiveStudentMLP` | **1,466,880** | **59.73%** |
| **Decoder State Proj** | Hidden & Cell Projections ($1024 \to 256$) | 524,800 | 21.37% |
| **Recurrent Cell** | LSTMCell ($2 \to 256$) | 266,240 | 10.84% |
| **Sequence Self-Attention** | Scaled Dot-Product ($Q, K, V$ Projections) | 197,376 | 8.04% |
| **Kinematic Output** | Linear ($256 \to 2$) | 514 | 0.02% |
| **Total Decoder Network** | `ContrastiveDecoder` | **988,930** | **40.27%** |
| **COMPLETE MODEL** | **Deployed Inference Network** | **2,455,810** | **100.00%** |

### Memory Footprint
- **FP32 Precision:** **9.82 MB**
- **FP16 / BF16 Precision:** **4.91 MB**
- **INT8 Quantization:** **2.46 MB**

---

## Academic Publication & Citation

The complete 17-page scientific journal paper is available in the [`paper/`](paper/) directory:
- **PDF:** [`paper/VJEPA_Grounded_Spatial_Representation_Journal.pdf`](paper/VJEPA_Grounded_Spatial_Representation_Journal.pdf)
- **LaTeX:** [`paper/VJEPA_Grounded_Spatial_Representation_Journal.tex`](paper/VJEPA_Grounded_Spatial_Representation_Journal.tex)
- **Markdown:** [`paper/VJEPA_Grounded_Spatial_Representation_Journal.md`](paper/VJEPA_Grounded_Spatial_Representation_Journal.md)

### BibTeX
```bibtex
@article{seger2026grounding,
  title={Grounding Spatial Representations via Visual World Models for Robust Autoregressive Drift Reduction in Low-Power Robot Telemetry},
  author={Seger, Mychal},
  year={2026}
}
```

---

## License
This repository is released under the [MIT License](LICENSE).
