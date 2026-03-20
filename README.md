# Information Capacity of EEG: Theoretical and Computational Limits of Recoverable Neural Information

**Author:** Ishir Rao, Yale University  
**Paper:** [arXiv:2510.17841](https://arxiv.org/abs/2510.17841)  
**Code:** `IT_EEG_Full_Code.ipynb`

---

## Overview

This repository contains the full simulation and analysis code accompanying the paper *"Information Capacity of EEG: Theoretical and Computational Limits of Recoverable Neural Information."*

The project asks a precise quantitative question: **how much information about cortical neural activity can EEG recordings actually carry?** To answer it, the code combines:

- **Analytic information theory** — deriving closed-form mutual information bounds for Gaussian channels
- **Synthetic forward modeling** — simulating realistic EEG measurements with controlled noise and spatial blurring
- **Empirical estimation** — using k-nearest-neighbor (KSG) mutual information estimators on simulated data
- **Decoder benchmarking** — comparing linear (Ridge) and nonlinear (MLP) reconstruction of latent neural sources
- **Information Bottleneck (IB) analysis** — training variational IB models to trace the compression–reconstruction tradeoff
- **Animated visualization** — rendering a dynamic topographic EEG simulation across time

The central finding is that scalp EEG is a **fundamentally narrow channel**: it conveys only tens of bits per sample about low-dimensional cortical activity, saturates with ~64–128 electrodes, and is bottlenecked primarily by signal-to-noise ratio (SNR) rather than sensor density or decoder sophistication.

---

## Repository Structure

```
IT_EEG_Full_Code.ipynb     # Single notebook containing all code (3 major cells)
README.md                  # This file
output_eeg_upgraded/       # Auto-generated on run (figures, CSVs, summary)
```

The notebook is self-contained. Running it top-to-bottom produces all results, figures, and output files reported in the paper.

---

## Notebook Structure

The notebook is organized into three major cells:

### Cell 1 — Forward Modeling, Analytic MI, and Decoder Benchmarking

This is the primary simulation cell. It implements the full synthetic pipeline:

**Generative model setup:**
- Simulates `n_sources = 64` cortical sources arranged on a circle, whose spatial covariance follows a radial basis function (RBF) kernel
- Projects source activity through `n_latents = 8` dominant eigenmodes of the source covariance
- Drives latents with AR(1) autoregressive dynamics (coefficient `ρ = 0.9`) to model temporal autocorrelation
- Constructs a Gaussian spatial blur leadfield matrix `A` mapping sources to electrodes
- Adds spatially correlated Gaussian noise `ε` with covariance `Σ_ε`

**Analytic mutual information:**

For jointly Gaussian signals, the mutual information between cortical latents and EEG sensors is computed analytically using:

```
I(X; Y) = (1/2) * log2 det(I + A Σ_X Aᵀ Σ_ε⁻¹)
```

This is evaluated across all combinations of:
- **Electrode counts:** 8, 16, 32, 64, 128
- **SNR levels:** 0 dB, 10 dB, 20 dB

**Empirical mutual information:**

MI is also estimated empirically using the **KSG (Kraskov–Stögbauer–Grassberger) k-nearest-neighbor estimator** between PCA-reduced EEG data and true latent states. This validates the analytic bound and quantifies finite-sample losses.

**Decoding:**

Two decoders are trained to reconstruct latent source activity from EEG:
- **Ridge regression** (linear)
- **MLP** (one hidden layer, 128 units, ReLU, 400 epochs)

Performance is reported as variance-weighted R² and as mutual information between true and predicted latents (also via KSG).

**Outputs:**
- `results_table.csv` — all analytic MI, empirical MI, R², and decoder MI values
- `figA_schematic.png` — spatial layout of sources and electrodes
- `fig1_mi_vs_electrodes.png` — analytic vs empirical MI as a function of electrode count
- `fig2_mi_vs_snr.png` — MI as a function of SNR for selected electrode densities
- `fig3_decoder_vs_mi.png` — decoder R² vs analytic MI
- `fig4_decoder_mi.png` — decoder-recovered MI vs analytic MI bound
- `run_summary.txt` — elapsed time and descriptive statistics

---

### Cell 2 — Variational Information Bottleneck (VIB) Analysis

This cell extends the analysis using the **Information Bottleneck framework**, which explores the tradeoff between how much information a compressed representation `T` retains about the EEG input `Y` (compression) vs. how much it preserves about the original latent sources `X` (relevance).

**Model architecture:**

A variational encoder-decoder is trained for each `(n_elec, snr_db)` pair:
- **Encoder:** 2-layer MLP → produces `μ_z` and `log σ²_z` for a `z_dim = 16` Gaussian bottleneck
- **Reparameterization:** `z = μ_z + ε · σ_z` (standard VAE trick)
- **Decoder:** 2-layer MLP → reconstructs latent sources `X`

The loss is:

```
L = reconstruction loss + β · KL(q(z|Y) || p(z))
```

By sweeping `β ∈ {0.001, 0.01, 0.1, 1, 10}`, the code traces the full IB tradeoff curve for each experimental condition.

**Quantities estimated:**
- `I(T; Y)` — compression (estimated via KL divergence)
- `I(X; T)` — relevance (proxy via decoder MSE)

**Outputs:**
- `ib_results_{n_elec}_{snr_db}.csv` — per-condition IB results for each β
- `ib_summary_all.csv` — combined IB results across all conditions
- `fig_IB_grid.png` — grid of IB tradeoff curves for each (electrode count × SNR) combination

---

### Cell 3 — Animated EEG Simulation

This cell (credited to Google Gemini) produces a dynamic visualization of the forward model. It simulates a simplified EEG system in real time and saves an animation showing how latent neural activity propagates through sources and sensors.

**Simulation parameters:**
- 64 cortical sources, 8 latent AR(1) processes, 32 scalp electrodes
- AR(1) coefficient `ρ = 0.9`, 400 time steps, SNR = 10 dB

**Animation panels:**
- **Top:** Topographic sensor map — electrode positions colored by instantaneous EEG amplitude, shown inside a head outline
- **Bottom left:** Rolling traces of 6 example source signals
- **Bottom right:** Rolling traces of all 8 latent processes

**Output:**
- `eeg_sim_animation.gif` — saved to `/mnt/data/`

---

## Key Parameters

All simulation parameters are defined at the top of Cell 1 and can be adjusted:

| Parameter | Default | Description |
|---|---|---|
| `n_sources` | 64 | Number of cortical sources |
| `n_latents` | 8 | Latent dimensionality to evaluate |
| `n_time` | 2000 | Number of time samples (reduce to 1000 for speed) |
| `electrode_list` | [8,16,32,64,128] | Electrode counts to sweep |
| `snr_db_list` | [0, 10, 20] | SNR values to sweep (dB) |
| `ar_rho` | 0.9 | AR(1) temporal autocorrelation coefficient |
| `leadfield_blur` | 0.25 | Spatial spread of the leadfield (Gaussian σ) |
| `blur_sigma_sources` | 0.15 | Source field smoothness (RBF kernel σ) |
| `noise_spatial_sigma` | 0.3 | Spatial correlation length of sensor noise |

---

## Requirements

### Python packages

```
numpy
scipy
scikit-learn
matplotlib
pandas
torch          # for Cell 2 (VIB)
pillow         # for Cell 3 (GIF animation)
```

Install with:

```bash
pip install numpy scipy scikit-learn matplotlib pandas torch pillow
```

### Hardware

- **CPU** is sufficient for Cells 1 and 3
- Cell 2 (VIB) will use a **GPU automatically** if available via PyTorch (`cuda`), but runs fine on CPU for the default settings

---

## Running the Code

1. Clone the repository and install dependencies (see above)
2. Open `IT_EEG_Full_Code.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells in order (Kernel → Restart & Run All)
4. Outputs are saved to `output_eeg_upgraded/` (created automatically)

> **Note:** Cell 1 is the most compute-intensive. With default settings (`n_time = 2000`, all 5 electrode counts × 3 SNR levels), it runs in a few minutes on a modern CPU. Reduce `n_time` to 1000 for faster iteration.

---

## Main Results

| Finding | Details |
|---|---|
| **Information capacity is low** | Analytic MI reaches only ~10–40 bits/sample depending on SNR, even under idealized Gaussian assumptions |
| **Saturation with electrode count** | MI growth flattens around 64–128 electrodes; additional sensors provide diminishing returns |
| **SNR is the primary bottleneck** | A 10 dB SNR increase yields a 2–3× gain in recoverable bits — far more than doubling electrode count |
| **Linear decoders nearly suffice** | Ridge and MLP decoders achieve R² ≈ 0.85–0.9; the MLP offers little additional benefit, confirming the forward model is approximately linear |
| **Decoder MI gap** | Decoder-recovered MI is only ~1/6 of the analytic bound — the loss is due to measurement physics (blurring + noise), not algorithmic limitations |

---

## Citation

If you use this code or build on these results, please cite the accompanying paper:

```
@article{rao2025eeg,
  title={Information Capacity of EEG: Theoretical and Computational Limits of Recoverable Neural Information},
  author={Rao, Ishir},
  journal={arXiv preprint arXiv:2510.17841},
  year={2025}
}
```

---

## References

1. Cover, T. M. & Thomas, J. A. *Elements of Information Theory.* Wiley, 1991.
2. Shannon, C. E. "A Mathematical Theory of Communication." *Bell Syst. Tech. J.*, 1948.
3. Nunez, P. L. & Srinivasan, R. *Electric Fields of the Brain.* Oxford University Press, 2006.
4. Goldenholz, D. M. et al. "Mapping the signal-to-noise ratios of cortical sources in MEG and EEG." *Clin. Neurophysiol.*, 2009.
5. Grech, R. et al. "Review on solving the inverse problem in EEG source analysis." *Clin. Neurophysiol.*, 2008.
6. Panzeri, S. et al. "Neural coding and decoding: Theoretical principles and practical strategies." *Neuron*, 2017.
7. Kraskov, A., Stögbauer, H. & Grassberger, P. "Estimating mutual information." *Phys. Rev. E*, 2004.
