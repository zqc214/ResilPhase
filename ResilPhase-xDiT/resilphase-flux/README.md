# ResilPhase-FLUX-xDiT

This package adapts `ResilPhase-Diffusers/resilphase-flux` to xDiT/xFuser FLUX
inference.

It keeps the ResilPhase-FLUX algorithm unchanged:

- full steps run FLUX double-stream and single-stream transformer blocks;
- full steps cache whole-stack deltas for image double stream, text double stream,
  and concatenated single stream;
- cache steps predict those deltas with phase-axis barycentric Lagrange
  interpolation.

## Install

```bash
cd /mnt/public/zqc/ResilPhase/ResilPhase-xDiT/resilphase-flux
pip install -e .
```

## Run

```bash
bash scripts/run_flux_xdit.sh
```

The script defaults to:

- `MODEL_ID=/root/autodl-tmp/black-forest-labs/FLUX.1-dev`
- `N_GPUS=8`
- `INFERENCE_STEP=50`
- `PROMPT="A dog standing on the grass."`

You can override them:

```bash
N_GPUS=4 MODEL_ID=/path/to/FLUX.1-dev PROMPT="A photo of a red car" bash scripts/run_flux_xdit.sh
```

## Notes

This adapter uses `xFuserFluxPipeline` and patches the xFuser FLUX transformer
wrapper at runtime. It follows the same integration pattern as the TaylorSeer
xDiT FLUX example, but replaces Taylor extrapolation with ResilPhase phase-axis
Lagrange interpolation.

The forward path is written to tolerate both current Diffusers FLUX block
signatures and older xFuser single-block signatures.
