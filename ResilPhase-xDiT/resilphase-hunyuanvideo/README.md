# ResilPhase-HunyuanVideo-xDiT

This package adapts `ResilPhase-Diffusers/resilphase-hunyuanvideo` to an
xDiT/xFuser-style inference entry point.

It keeps the ResilPhase-HunyuanVideo algorithm unchanged:

- full calls run all double-stream and single-stream transformer blocks;
- full calls cache whole-stack deltas for video/token double blocks and
  concatenated single blocks;
- cache calls predict those deltas with phase-axis barycentric Lagrange
  interpolation.

## Current xFuser Scope

The xFuser version installed in this environment does not provide a registered
wrapper for Diffusers `HunyuanVideoPipeline` or `HunyuanVideoTransformer3DModel`.
This project adds a minimal `xFuserHunyuanVideoPipeline` registration so the run
path can use xFuser configuration and `torchrun`.

Supported:

- xFuser CLI/config initialization;
- Diffusers `HunyuanVideoPipeline` execution;
- ResilPhase patching;
- data parallel splitting over prompt lists.

Not supported by this adapter:

- sequence parallelism;
- tensor parallelism;
- pipeline parallelism;
- xFuser CFG parallelism;
- parallel VAE.

Keep those degrees at `1`. Use `--data_parallel_degree` for multi-GPU prompt
batches.

## Install

```bash
cd /mnt/public/zqc/ResilPhase/ResilPhase-xDiT/resilphase-hunyuanvideo
pip install -e .
```

## Run

```bash
bash scripts/run_hunyuanvideo_xdit.sh
```

Defaults:

- `MODEL_ID=hunyuanvideo-community/HunyuanVideo`
- `N_GPUS=1`
- `INFERENCE_STEP=50`
- `PROMPT="A cat walks on the grass, realistic style."`

Keep `--true-cfg-scale 1.0` unless cache state is extended to separate
conditional and unconditional transformer branches.
