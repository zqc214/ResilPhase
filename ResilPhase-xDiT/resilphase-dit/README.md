# ResilPhase-DiT-xDiT

This package adapts `ResilPhase-Diffusers/resilphase-dit` to an xDiT/xFuser-style
inference entry point.

It keeps the ResilPhase-DiT algorithm unchanged:

- full calls run all `DiTTransformer2DModel` transformer blocks;
- full calls cache the whole block-stack delta;
- cache calls skip all transformer blocks and predict that delta with phase-axis
  barycentric Lagrange interpolation.

## Current xFuser Scope

The xFuser version installed in this environment does not provide a registered
wrapper for Diffusers `DiTPipeline` or `DiTTransformer2DModel`. This project adds
a minimal `xFuserDiTPipeline` registration so the run path can use xFuser
configuration and `torchrun`.

Supported:

- xFuser CLI/config initialization;
- Diffusers `DiTPipeline` execution;
- ResilPhase patching;
- data parallel splitting over `class_labels`.

Not supported by this adapter:

- sequence parallelism;
- tensor parallelism;
- pipeline parallelism;
- xFuser CFG parallelism;
- parallel VAE.

Keep those degrees at `1`. Use `--data_parallel_degree` for multi-GPU batches.

## Install

```bash
cd /mnt/public/zqc/ResilPhase/ResilPhase-xDiT/resilphase-dit
pip install -e .
```

## Run

Single GPU:

```bash
N_GPUS=1 bash scripts/run_dit_xdit.sh
```

Data parallel, with one or more class labels:

```bash
N_GPUS=2 CLASS_LABELS=495,207 bash scripts/run_dit_xdit.sh
```

Direct command:

```bash
torchrun --nproc_per_node=1 examples/dit_class_conditional_xdit.py \
  --model facebook/DiT-XL-2-256 \
  --class-labels 495 \
  --num_inference_steps 50 \
  --guidance_scale 4.0 \
  --data_parallel_degree 1 \
  --pipefusion_parallel_degree 1 \
  --ulysses_degree 1 \
  --ring_degree 1 \
  --tensor_parallel_degree 1
```

Outputs are saved under `./results` by default.
