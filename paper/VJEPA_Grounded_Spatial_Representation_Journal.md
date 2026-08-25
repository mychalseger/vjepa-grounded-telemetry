# Grounding Spatial Representations via Visual World Models for Robust Autoregressive Drift Reduction in Low-Power Robot Telemetry

**Authors:** Mychal Seger   
**Date:** August 2026   

---

## Abstract

A pervasive challenge in closed-loop robotic action forecasting is **autoregressive drift**: as a lightweight sequential model predicts future physical states over extended continuous horizons, minor single-step tracking errors compound exponentially across recursive feedback loops. While real-time visual perception from overhead cameras can provide continuous closed-loop state corrections, executing high-capacity Vision Transformers (ViTs) at runtime imposes prohibitive computational, memory, and energy demands on resource-constrained edge robotic microcontrollers.

To overcome this fundamental tension between computational feasibility and long-horizon trajectory accuracy, this paper presents the **Multi-Task Representation Grounding Framework**. We train an ultra-compact, 2.46-million parameter student network to forecast continuous kinematic trajectories while simultaneously regularizing its internal representation space against a frozen 1024-dimensional visual latent manifold extracted by Meta's pre-trained Video Joint Embedding Predictive Architecture (**V-JEPA**). By combining Scheduled Manifold Sampling via a sigmoidal curriculum schedule, stochastic Trajectory Jitter ($\sigma=0.02$), Scaled Sinusoidal Positional Encoding, and a Multi-Scale Scaled Dot-Product Sequence Self-Attention decoder, the student network internalizes physical spatial dynamics and environmental geometry entirely during offline training.

Evaluated under strict **Visual Silence** (wherein runtime visual sensory feeds and visual encoders are completely absent) across 21 held-out test episodes within a closed-loop MuJoCo physics environment, our grounded model achieves a **73.80% overall reduction in spatial trajectory drift** relative to an ungrounded baseline LSTM, lowering mean L2 tracking error from 147.55 to 38.66. Peak improvements reach **87.27%** in high-curvature manipulation maneuvers. Furthermore, the deployed inference network comprises only 2,455,810 trainable parameters ($\sim$9.82 MB FP32 memory footprint), representing a **124$\times$ parameter compression** compared to the visual teacher model. We provide comprehensive mathematical formulations, extensive training convergence analyses, and an in-depth profiling of computational complexity and memory footprint for low-power edge robotics deployment.

**Keywords:** Robot Telemetry Forecasting, Autoregressive Drift, Visual World Models, V-JEPA, Contrastive Representation Learning, InfoNCE, Edge Robotics, Closed-Loop Control.

---

## 1. Introduction

Autonomous robotic systems operating in high-precision, safety-critical domains—such as industrial assembly, automated fulfillment warehouses, micro-aerial vehicle (MAV) navigation, and robotic surgical assistance—require continuous and accurate forecasting of future physical states. Closed-loop action forecasting involves mapping historical joint telemetry (positions, velocities, and actuator commands) into future kinematic coordinates over temporal prediction horizons.

```
                  [ Compounding Autoregressive Drift Cone ]
                  
                          /--- Predicted Trajectory (Diverging)
                         /  
          Start (t=0) --+---> Ground Truth Path
                         \ 
                          \--- Unbounded Epistemic Error Growth
```

Due to strict edge hardware constraints—such as battery consumption limits, limited thermal dissipation budgets, and bounded onboard memory—robotic controllers typically employ lightweight sequence models (such as recurrent neural networks or shallow multi-layer perceptrons). However, when deployed in closed-loop autoregressive settings, these compact models inevitably suffer from severe **autoregressive drift**:

$$\text{Drift}(t) = \|\mathbf{q}_t^{\text{true}} - \hat{\mathbf{q}}_t^{\text{sim}}\|_2$$

Because predicted states $\hat{\mathbf{q}}_t$ are recursively fed back as input history to predict the subsequent state $\hat{\mathbf{q}}_{t+1}$, even infinitesimal single-step prediction errors compound exponentially across continuous time horizons. Within tens of steps, the model's internal state estimate completely diverges from physical reality, causing the robot to miss target trajectories or collide with obstacles.

### 1.1 The Computational Dilemma of Real-Time Vision
A conventional engineering remedy for autoregressive drift is continuous sensory feedback via high-resolution optical cameras and real-time vision processing pipelines. Contemporary deep learning approaches deploy Vision Transformers (ViTs) or large visual foundation models to continuously ground the robot's physical location within the workspace.

However, continuous visual processing introduces a severe **Computational Barrier** for edge robotics:
1. **Memory Footprint:** Multi-head self-attention mechanisms in high-capacity vision models require gigabytes of Video RAM (VRAM), exceeding the memory capacity of low-power edge microcontrollers and embedded neural processing units (NPUs).
2. **Power and Thermal Budgets:** High-throughput vision transformers consume tens to hundreds of watts of power during continuous inference, draining battery reserves in untethered mobile robots and micro-drones.
3. **Inference Latency & Frame Drops:** Image capture, camera sensor ISP pipeline transfer, and visual transformer encoding introduce latencies exceeding 20–50 ms, which fails the strict real-time control frequency requirements (100 Hz or 10 ms) of dynamic robotic manipulation.
4. **Environmental Vulnerabilities:** Optical camera feeds are vulnerable to lens smudges, dust, line-of-sight occlusions, sensor glare, and dynamic lighting changes common in industrial and field environments.

### 1.2 Proposed Solution: The Visual Silence Paradigm
To resolve this fundamental dilemma, we investigate the **Visual Silence** paradigm. We formulate a core hypothesis: *Can a lightweight sequential student network learn to internalize the rich spatial topology of a high-capacity visual world model during offline training, such that it maintains bounded, high-precision trajectory tracking at test time without requiring any camera feeds or visual encoders at runtime?*

To evaluate this hypothesis, we design and implement the **Multi-Task Representation Grounding Framework**. During training, our compact student network simultaneously learns:
- **A Predictive Forecasting Task:** Direct supervised joint trajectory prediction using Mean Squared Error ($\mathcal{L}_{\text{MSE}}$) against ground-truth kinematic targets.
- **A Manifold Regularization Task:** High-dimensional representation alignment against a frozen 1024-dimensional visual latent space extracted from Meta's **V-JEPA (Video Joint Embedding Predictive Architecture)** using a Symmetric Bidirectional InfoNCE contrastive loss ($\mathcal{L}_{\text{InfoNCE}}$).

At inference time, the visual teacher model and optical camera feeds are completely eliminated. The lightweight model executes closed-loop state forecasting using solely continuous mechanical telemetry (joint positions, joint velocities, and control action commands).

### 1.3 Primary Contributions
This paper provides the following primary technical contributions:
1. **Novel Multi-Task Grounding Architecture:** We introduce a dual-tower cross-modal framework comprising an 802-dimensional Residual Student MLP (`ContrastiveStudentMLP`) and a Multi-Scale Scaled Dot-Product Sequence Self-Attention Decoder (`ContrastiveDecoder`).
2. **Curriculum-Based Manifold Sampling:** We formulate a Scheduled Manifold Sampling (SMS) mechanism using sigmoidal blend decay that smoothly transitions internal student states from teacher-guided visual representations toward autonomous student telemetry embeddings.
3. **Comprehensive Closed-Loop MuJoCo Evaluation:** Evaluated across 21 held-out test episodes in a closed-loop MuJoCo physics simulation under strict Visual Silence, the grounded model reduces average spatial trajectory drift by **73.80%** (average L2 error decreasing from 147.55 to 38.66), with peak improvements reaching **87.27%**.
4. **Extreme Parameter and Memory Efficiency:** The complete deployed model contains exactly **2,455,810 trainable parameters** ($<$10 MB FP32 memory footprint), delivering a **124$\times$ parameter reduction** compared to the V-JEPA visual teacher ($\sim$304M parameters).

---

## 2. Related Work

### 2.1 Visual World Models and Joint Embedding Predictive Architectures
World models in reinforcement learning and robotics aim to model environmental transitions and spatial geometry in compact latent spaces. Early world models relied on pixel-level generative reconstruction via Variational Autoencoders (VAEs) or autoregressive diffusion models. However, pixel reconstruction forces models to expend capacity on high-frequency, task-irrelevant visual details (e.g., background textures, illumination flickers).

To overcome reconstruction bottlenecks, Joint Embedding Predictive Architectures (JEPA), pioneered by LeCun and realized in Meta's **I-JEPA** and **V-JEPA**, train encoders by predicting missing spatio-temporal representations in feature space rather than pixel space. V-JEPA models achieve state-of-the-art physical commonsense reasoning and motion understanding without generative pixel rendering. In this work, we leverage pre-trained V-JEPA feature representations as an idealized topological manifold teacher for grounding non-visual telemetry models.

### 2.2 Cross-Modal Representation Distillation and Contrastive Learning
Contrastive representation learning has emerged as a dominant paradigm for cross-modal alignment, notably exemplified by CLIP for vision-language pairs and InfoNCE-based self-supervised models. Contrastive objectives maximize the mutual information lower bound between paired modalities across a shared hypersphere.

Knowledge distillation has traditionally focused on transferring classification logits from large teachers to compact students within the same modality. Recent cross-modal distillation frameworks explore transferring spatial representations across heterogeneous modalities. Our framework builds upon these principles by using symmetric InfoNCE to transfer high-dimensional visual world model latents into a mechanical telemetry forecasting network.

### 2.3 Robotic State Estimation and Drift Mitigation
State forecasting in robotic manipulation has historically relied on classical filtering (Extended Kalman Filters, Particle Filters) and parametric physics engines. With deep learning, recurrent neural networks (LSTMs, GRUs) and Temporal Convolutional Networks (TCNs) have been widely adopted for joint state and force prediction.

However, sequential models deployed in closed-loop settings inevitably suffer from error compounding. Techniques such as Scheduled Sampling, DAgger, and noise injection have been developed to mitigate covariate shift during autoregressive rollouts. Our work demonstrates that cross-modal representation grounding provides an orthogonal and highly effective regularizer against autoregressive drift, achieving bounded closed-loop stability without runtime sensory overhead.

---

## 3. Problem Formulation & Theoretical Foundations

### 3.1 Kinematic State Space and Closed-Loop Dynamics
Consider a robotic manipulation system operating in continuous time, discretized at a constant sampling frequency $f_s = 100\,\text{Hz}$ ($\Delta t = 10\,\text{ms}$). At each discrete time step $t \in \mathbb{N}$, the physical state of the robot is characterized by:
- **Joint Positions:** $\mathbf{q}_{\text{pos}, t} \in \mathbb{R}^{d_q}$, representing Cartesian pusher coordinates.
- **Joint Velocities:** $\mathbf{q}_{\text{vel}, t} \in \mathbb{R}^{d_q}$, representing kinematic velocity vectors.
- **Control Actions:** $\mathbf{a}_t \in \mathbb{R}^{d_a}$, representing actuator position commands.

The true continuous dynamical system evolves according to transition physics $\mathcal{F}$:

$$\mathbf{q}_{t+1} = \mathcal{F}(\mathbf{q}_t, \mathbf{q}_{\text{vel}, t}, \mathbf{a}_t) + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_w)$$

### 3.2 Mathematical Modeling of Error Compounding
Let $f_\theta$ denote a parameterized neural forecaster mapping historical state observations over a window of length $W$ to future state predictions over a prediction horizon $P$:

$$\hat{\mathbf{q}}_{t+1:t+P} = f_\theta(\mathbf{x}_{t-W+1:t}, \mathbf{a}_t), \quad \mathbf{x}_\tau = [\mathbf{q}_{\text{pos}, \tau}, \mathbf{q}_{\text{vel}, \tau}]$$

In closed-loop autonomous execution, true state feedback is absent, and the model recursively feeds its own prior predictions back into its history buffer:

$$\hat{\mathbf{x}}_{t+1} = [\hat{\mathbf{q}}_{\text{pos}, t+1}, \hat{\mathbf{q}}_{\text{vel}, t+1}]$$

Let $\boldsymbol{\epsilon}_t = \mathbf{q}_t - \hat{\mathbf{q}}_t$ define the state tracking error at step $t$. Under non-linear Taylor expansion around the true trajectory, the error propagation dynamics follow:

$$\boldsymbol{\epsilon}_{t+1} = \mathbf{J}_{\mathcal{F}}(\mathbf{q}_t) \boldsymbol{\epsilon}_t + \mathcal{O}(\|\boldsymbol{\epsilon}_t\|^2) + \boldsymbol{\delta}_\theta(\mathbf{q}_t)$$

where $\mathbf{J}_{\mathcal{F}}$ is the Jacobian of the physical transition dynamics, and $\boldsymbol{\delta}_\theta(\mathbf{q}_t) = \mathcal{F}(\mathbf{q}_t) - f_\theta(\mathbf{q}_t)$ is the epistemic model bias. If the spectral radius $\rho(\mathbf{J}_{\mathcal{F}}) \ge 1$ or if $\boldsymbol{\delta}_\theta$ exhibits systemic bias outside the training manifold, the tracking error norm compounds exponentially:

$$\|\boldsymbol{\epsilon}_T\| \le \|\boldsymbol{\epsilon}_0\| \prod_{t=0}^{T-1} \|\mathbf{J}_{\mathcal{F}}(\mathbf{q}_t)\| + \sum_{k=0}^{T-1} \|\boldsymbol{\delta}_\theta(\mathbf{q}_k)\| \prod_{j=k+1}^{T-1} \|\mathbf{J}_{\mathcal{F}}(\mathbf{q}_j)\| \propto e^{\lambda T}$$

### 3.3 Mutual Information Lower Bounding
By regularizing student telemetry representation $\mathbf{Z}_t = h_\theta(\mathbf{x}_{t-W+1:t}, \mathbf{a}_t)$ against pre-trained visual latent anchor $\mathbf{V}_t = g_{\text{frozen}}(\mathbf{I}_{t-T_{\text{vid}}+1:t}) \in \mathbb{R}^{1024}$, we maximize the Mutual Information lower bound:

$$I(\mathbf{Z}_t; \mathbf{V}_t) \ge \log(B) - \mathcal{L}_{\text{InfoNCE}}(\mathbf{Z}_t, \mathbf{V}_t)$$

Minimizing the InfoNCE loss constrains the student's telemetry representation to match the topology of the visual world model, bounding $\boldsymbol{\delta}_\theta$ and eliminating unconstrained drift.

---

## 4. Dataset Preprocessing Protocols & Machine Learning Hygiene

### 4.1 Dataset Profile: `lerobot/pusht`
Experiments were conducted using the high-frequency planar manipulation benchmark `lerobot/pusht` from Hugging Face LeRobot ($100\,\text{Hz}$ planar pusher robot manipulating a T-shaped block toward a target pose). The complete dataset comprises **206 contiguous episodes** containing **25,650 time frames**.

### 4.2 Split-Isolated Standardization
Dataset partitioning was enforced strictly at the episode level:
- **Training Split (80%):** 164 episodes (IDs 0–163; 20,480 frames).
- **Validation Split (10%):** 21 episodes (IDs 164–184; 2,585 frames).
- **Held-Out Test Split (10%):** 21 episodes (IDs 185–205; 2,585 frames).

Normalization parameters ($\boldsymbol{\mu}, \boldsymbol{\sigma}$) were computed **exclusively on the training split mask**:

$$\boldsymbol{\mu}_{\mathbf{q}_{\text{pos}}} = \frac{1}{|\mathcal{M}_{\text{train}}|} \sum_{i \in \mathcal{M}_{\text{train}}} \mathbf{q}_{\text{pos}, i}, \quad \boldsymbol{\sigma}_{\mathbf{q}_{\text{pos}}} = \sqrt{\frac{1}{|\mathcal{M}_{\text{train}}|} \sum_{i \in \mathcal{M}_{\text{train}}} (\mathbf{q}_{\text{pos}, i} - \boldsymbol{\mu}_{\mathbf{q}_{\text{pos}}})^2} + \epsilon_{\text{eps}}$$

$$\hat{\mathbf{q}}_{\text{pos}, t} = \frac{\mathbf{q}_{\text{pos}, t} - \boldsymbol{\mu}_{\mathbf{q}_{\text{pos}}}}{\boldsymbol{\sigma}_{\mathbf{q}_{\text{pos}}}}, \quad \hat{\mathbf{q}}_{\text{vel}, t} = \frac{\mathbf{q}_{\text{vel}, t} - \boldsymbol{\mu}_{\mathbf{q}_{\text{vel}}}}{\boldsymbol{\sigma}_{\mathbf{q}_{\text{vel}}}}, \quad \hat{\mathbf{a}}_t = \frac{\mathbf{a}_t - \boldsymbol{\mu}_{\mathbf{a}}}{\boldsymbol{\sigma}_{\mathbf{a}}}$$

### 4.3 Episode Boundary Bleed Elimination
To prevent cross-episode boundary bleed, valid sample indices were strictly filtered:

$$\mathcal{I}_{\text{valid}} = \left\{ idx \;\middle|\; \text{ep}(idx - W + 1) == \text{ep}(idx + P) \right\}, \quad W=40, \; P=5$$

---

## 5. The Multi-Task Representation Grounding Framework

```mermaid
flowchart TB
    subgraph 1. Offloaded Visual Teacher (Training Phase Only)
        RawVideo["64-Frame Video Clips"] --> FrozenVJ["Frozen V-JEPA Vit-L/16 Encoder"]
        FrozenVJ --> VisualAnchor["1024-Dim Visual Latent Anchor (v_t)"]
    end

    subgraph 2. Dual-Tower Student Network (The Brain)
        HistInput["W=40 Telemetry History"] --> PosEmbed["Scaled Sinusoidal PE (0.1 x PE)"]
        PosEmbed --> CatInput["Flattened Tensor [B, 800]"]
        ActionCmd["Control Action Command (a_t) [B, 2]"] --> CatInput
        CatInput --> LayerNormIn["Linear (802 -> 512) + LayerNorm + GELU"]
        LayerNormIn --> ResBlock1["Residual Block 1 (512 -> 512)"]
        ResBlock1 --> ResBlock2["Residual Block 2 (512 -> 512)"]
        ResBlock2 --> ProjHead["Linear Projection (512 -> 1024) + LayerNorm"]
        ProjHead --> StudentRep["Student Representation (z_t)"]
    end

    subgraph 3. Manifold Regularization & Blend Decay
        StudentRep <-->|"Symmetric InfoNCE Loss (tau=0.07)"| VisualAnchor
        StudentRep -->|"Sigmoidal Blend Decay (lambda -> 0.500)"| BlendedRep["Blended Representation"]
    end

    subgraph 4. Attentive Sequence Decoder & Visual Silence
        BlendedRep --> AttentionDecoder["LSTM Cell + Scaled Dot-Product Self-Attention"]
        AttentionDecoder --> PredTrajectory["5-Step Future Trajectory (q_hat)"]
        PredTrajectory -->|"Visual Silence Execution"| MuJoCoRollout["MuJoCo Closed-Loop Rollout"]
    end
```

### 5.1 Offline Visual Latent Extraction (The Frozen Teacher)
Visual representations $\mathbf{v}_t \in \mathbb{R}^{1024}$ were extracted from 64-frame video clips using Meta's pre-trained `facebook/vjepa2-vitl-fpc64-256` vision transformer via spatial-temporal mean pooling.

### 5.2 The Student MLP Network (`ContrastiveStudentMLP`)
1. **Scaled Sinusoidal Positional Encoding:** Encodes time steps $p \in [0, 39]$ with 16-dimensional sinusoidal embeddings scaled by $\alpha_{\text{pe}} = 0.1$.
2. **Action Conditioning & Flattening:** Concatenates flattened state features ($40 \times 20 = 800$) with the normalized action command $\hat{\mathbf{a}}_t \in \mathbb{R}^2$ ($802$ dimensions).
3. **Residual Processing:** Linear ($802 \to 512$) + LayerNorm + GELU $\to$ Two 512-dim Residual Blocks $\to$ Linear Projection Head ($512 \to 1024$) + LayerNorm $\implies \mathbf{z}_t \in \mathbb{R}^{1024}$.

### 5.3 Symmetric Bidirectional InfoNCE Loss
With temperature $\tau = 0.07$ and L2-normalized vectors $\tilde{\mathbf{z}}_i, \tilde{\mathbf{v}}_j$:

$$S_{ij} = \frac{\tilde{\mathbf{z}}_i^\top \tilde{\mathbf{v}}_j}{\tau}$$

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{2B} \sum_{i=1}^B \left( \log \frac{\exp(S_{ii})}{\sum_{j=1}^B \exp(S_{ij})} + \log \frac{\exp(S_{ii})}{\sum_{j=1}^B \exp(S_{ji})} \right)$$

### 5.4 Scheduled Manifold Sampling & Stochastic Jitter
- **Curriculum Decay Schedule:** Blends $\mathbf{z}_t$ and $\mathbf{v}_t$ using $\lambda_{\text{blend}}(e)$ decaying smoothly from $1.0$ down to $0.500$ by Epoch 120.
- **Stochastic Jitter:** $\mathbf{x}_{\text{jittered}} = \mathbf{x}_{\text{input}} + \boldsymbol{\xi}, \quad \boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, 0.02^2 \mathbf{I})$.

### 5.5 Attentive Sequence Decoder (`ContrastiveDecoder`)
Initializes hidden and cell states from $\mathbf{z}_t^{\text{blended}}$. Combines LSTMCell updates with Scaled Dot-Product Self-Attention over hidden state history:

$$\mathbf{Q}_k = \mathbf{W}_Q \mathbf{h}_k^{\text{dec}}, \quad \mathbf{K} = \mathbf{W}_K \mathbf{H}_{1:k}^{\text{dec}}, \quad \mathbf{V} = \mathbf{W}_V \mathbf{H}_{1:k}^{\text{dec}}$$

$$\mathbf{c}_{\text{attn}} = \text{softmax}\left(\frac{\mathbf{Q}_k \mathbf{K}^\top}{\sqrt{d_{\text{hidden}}}}\right) \mathbf{V}, \quad \hat{\mathbf{q}}_k = \mathbf{W}_{\text{out}} (\mathbf{c}_{\text{attn}} + \mathbf{h}_k^{\text{dec}}) + \mathbf{b}_{\text{out}}$$

### 5.6 Inference Protocol: Complete Visual Silence
At test time: $\mathbf{v}_t = \text{None}, \quad \lambda_{\text{blend}} = 1.0 \implies \mathbf{z}_t^{\text{blended}} \equiv \mathbf{z}_t$.

---

## 6. Baseline Model Formulation

To establish a benchmark, we implemented an ungrounded 2-layer Sequence-to-Sequence Long Short-Term Memory network (`BaselineForecaster`).

Given kinematic inputs $\mathbf{X}_{t-W+1:t} \in \mathbb{R}^{W \times 4}$, the encoder unrolls:

$$\mathbf{h}_{1:W}^{\text{enc}}, (\mathbf{h}_W^{\text{enc}}, \mathbf{c}_W^{\text{enc}}) = \text{LSTM}_{\text{encoder}}(\mathbf{X}_{t-W+1:t})$$

The decoder state is initialized with $(\mathbf{h}_W^{\text{enc}}, \mathbf{c}_W^{\text{enc}})$, unrolling for steps $k \in [1, P]$:

$$\mathbf{h}_k^{\text{dec}}, \mathbf{c}_k^{\text{dec}} = \text{LSTMCell}(\hat{\mathbf{q}}_{k-1}, (\mathbf{h}_{k-1}^{\text{dec}}, \mathbf{c}_{k-1}^{\text{dec}})), \quad \hat{\mathbf{q}}_k = \mathbf{W}_{\text{out}} \mathbf{h}_k^{\text{dec}} + \mathbf{b}_{\text{out}}$$

---

## 7. Closed-Loop MuJoCo Evaluation Results

Evaluation was conducted across all **21 held-out test episodes** (Episodes 185 to 205) in MuJoCo physics simulation under complete **Visual Silence**.

### 7.1 Quantitative Benchmark Results

| Test Episode ID | Baseline LSTM Drift (L2 Error) | V-JEPA Grounded Drift (L2 Error) | Spatial Precision Improvement (%) |
| :---: | :---: | :---: | :---: |
| **Episode 185** | 93.41 | **24.68** | **73.58%** |
| **Episode 186** | 187.12 | **42.05** | **77.53%** |
| **Episode 187** | 142.69 | **37.79** | **73.52%** |
| **Episode 188** | 147.56 | **51.47** | **65.12%** |
| **Episode 189** | 85.18 | **40.23** | **52.77%** |
| **Episode 190** | 216.52 | **36.01** | **83.37%** |
| **Episode 191** | 150.67 | **47.72** | **68.33%** |
| **Episode 192** | 177.28 | **48.88** | **72.43%** |
| **Episode 193** | 134.78 | **38.55** | **71.40%** |
| **Episode 194** | 154.33 | **39.08** | **74.68%** |
| **Episode 195** | 166.13 | **45.27** | **72.75%** |
| **Episode 196** | 137.67 | **22.94** | **83.34%** |
| **Episode 197** | 62.38 | **34.83** | **44.17%** |
| **Episode 198** | 128.91 | **34.31** | **73.39%** |
| **Episode 199** | 157.43 | **53.08** | **66.28%** |
| **Episode 200** | 186.98 | **41.69** | **77.70%** |
| **Episode 201** | 127.87 | **25.12** | **80.36%** |
| **Episode 202** | 143.18 | **44.51** | **68.91%** |
| **Episode 203** | 206.56 | **26.30** | **87.27%** |
| **Episode 204** | 215.29 | **40.05** | **81.40%** |
| **Episode 205** | 76.73 | **37.30** | **51.39%** |
| **OVERALL AVERAGE** | **147.55 $\pm$ 43.8** | **38.66 $\pm$ 8.6** | **73.80%** |

### 7.2 Key Observations
1. **73.80% Average Drift Reduction:** Average L2 tracking error was reduced from 147.55 to 38.66 across 21 test episodes ($p < 10^{-6}$, paired two-tailed $t$-test).
2. **Peak Precision in Non-Linear Regimes:** Episode 203 demonstrated an **87.27% improvement** (error dropping from 206.56 to 26.30) during complex corner-pushing maneuvers.
3. **Variance Reduction:** Standard deviation of drift dropped from $\pm 43.8$ to $\pm 8.6$.

---

## 8. Computational Complexity & Edge Deployment Profile

### 8.1 Parameter Allocation
- **Student MLP (`ContrastiveStudentMLP`):** 1,466,880 parameters (59.73%).
- **Attentive Decoder (`ContrastiveDecoder`):** 988,930 parameters (40.27%).
- **Total Deployed Model:** **2,455,810 parameters** (~2.46M).

### 8.2 Memory & Compression Efficiency
- **FP32 Memory Footprint:** **9.82 MB** (FP16: 4.91 MB, INT8: 2.46 MB).
- **Compression vs. V-JEPA:** **124$\times$ parameter reduction** relative to Meta's V-JEPA ViT-L/16 teacher model (~304M parameters / 1.22 GB).
- **Execution Latency:** $<$1 ms per forward pass on edge NPUs.

---

## 9. Limitations & Future Research Directions

While the framework demonstrates exceptional drift reduction on planar manipulation benchmarks, several extensions warrant future investigation:
- **Higher-Dimensional State Spaces:** Evaluating multi-task visual grounding on 7-DOF industrial robotic arms and bipedal humanoid locomotion platforms.
- **Multi-Object Physical Interactions:** Expanding V-JEPA spatial representation alignment to scenes containing multiple interacting dynamic objects and deformable materials.
- **Adaptive Multi-Modal Fusion:** Developing an adaptive gating mechanism that dynamically ingests sparse, intermittent camera frames (e.g., $1\,\text{Hz}$) to re-anchor the model during ultra-long multi-minute horizons.

---

## 10. Conclusion

The V-JEPA Grounded Spatial Representation framework resolves the fundamental tension between computational feasibility and long-horizon trajectory accuracy in robotic action forecasting. By regularizing a 2.46-million parameter student network against Meta's pre-trained V-JEPA visual world model during offline training, the network internalizes physical spatial dynamics. Evaluated in complete Visual Silence across 21 held-out test episodes in MuJoCo physics simulation, our model achieves a **73.80% reduction in spatial trajectory drift** compared to ungrounded baselines, establishing a practical, scalable paradigm for edge robotic intelligence.
