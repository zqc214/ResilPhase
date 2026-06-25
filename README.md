# ResilPhase: Plug-and-Play Phase Mapping and Noise-Resilient Macro-Trajectory Extrapolation for Diffusion Acceleration

**Status:** Accepted by **ECCV 2026**

## Method Overview

**ResilPhase** is a diffusion acceleration method based on phase-axis interpolation.
The core idea is to reuse the temporal regularity of denoising trajectories by mapping diffusion steps onto a phase axis, storing features from selected full-computation steps, and predicting intermediate-step feature updates with interpolation in the mapped phase space.

This repository contains three ResilPhase implementations built on top of different diffusion backbones:

- **ResilPhase-DiT**: image generation on DiT
- **ResilPhase-FLUX**: image generation on FLUX.1
- **ResilPhase-HunyuanVideo**: video generation on HunyuanVideo

Depending on the backend, the current codebase supports one or both of the following phase mappings:

- **Balanced mapping**
- **Chebyshev-node mapping**

The corresponding interpolation and cache update logic is implemented inside each subproject's `resilphase_utils` module, while the method hyperparameters are controlled by each backend's cache initialization logic.

## Repository Structure

- [ResilPhase-DiT](/mnt/public/zqc/ResilPhase/ResilPhase-DiT)
  - Method README: [ResilPhase-DiT.md](/mnt/public/zqc/ResilPhase/ResilPhase-DiT/ResilPhase-DiT.md)
- [ResilPhase-FLUX](/mnt/public/zqc/ResilPhase/ResilPhase-FLUX)
  - Method README: [ResilPhase-FLUX.md](/mnt/public/zqc/ResilPhase/ResilPhase-FLUX/ResilPhase-FLUX.md)
- [ResilPhase-HunyuanVideo](/mnt/public/zqc/ResilPhase/ResilPhase-HunyuanVideo)
  - Method README: [ResilPhase-HunyuanVideo.md](/mnt/public/zqc/ResilPhase/ResilPhase-HunyuanVideo/ResilPhase-HunyuanVideo.md)

## Notes

- This top-level README is intended as the project overview.
- Each subproject keeps its own backend-specific setup and sampling instructions.
- If you want to release the repository publicly, replace the first-line placeholder with the exact paper title.
