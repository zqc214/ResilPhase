# Resilphase-HunyuanVideo-Diffusers

`Resilphase-HunyuanVideo-Diffusers` adapts the ResilPhase-HunyuanVideo
acceleration path to HuggingFace Diffusers HunyuanVideo pipelines.

It patches a loaded `HunyuanVideoPipeline` or `HunyuanVideoTransformer3DModel`
instance at runtime. It does not modify the installed `diffusers` package.

## Install

```bash
cd /mnt/public/zqc/ResilPhase/ResilPhase-Diffusers/resilphase-hunyuanvideo
pip install -e .
```

The distribution/package name is `Resilphase-HunyuanVideo-Diffusers`. The Python
import module is `resilphase_hunyuanvideo_diffusers`.

## Usage

```python
import torch
from diffusers import HunyuanVideoPipeline
from diffusers.utils import export_to_video
from resilphase_hunyuanvideo_diffusers import apply_resilphase

num_steps = 50

pipe = HunyuanVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

apply_resilphase(
    pipe,
    num_steps=num_steps,
    fresh_threshold=6,
    max_order=1,
    first_enhance=3,
    mapping_method="balanced",
    balance_alpha=0.55,
)

frames = pipe(
    prompt="A cat walks on the grass, realistic style.",
    num_inference_steps=num_steps,
    true_cfg_scale=1.0,
).frames[0]
export_to_video(frames, "resilphase_hunyuanvideo_diffusers.mp4", fps=15)
```

## Configuration

- `num_steps`: should match `num_inference_steps` for the current pipeline call.
- `fresh_threshold`: interval for full transformer computation.
- `first_enhance`: number of initial full-computation calls.
- `max_order`: interpolation order. The cache keeps `max_order + 1` full-computation samples.
- `mapping_method`: `balanced` or `chebyshev`.
- `balance_alpha`: phase-axis compression strength for `balanced` mapping.

## Current Scope

This patch targets `diffusers>=0.35.0` and `HunyuanVideoTransformer3DModel`.

The implementation follows the ResilPhase-HunyuanVideo design:

- full calls run all double-stream and single-stream transformer blocks;
- full calls cache whole-stack deltas for image/text double blocks and concatenated single blocks;
- cache calls predict those deltas with phase-axis barycentric Lagrange interpolation.

For this initial plugin, keep `true_cfg_scale=1.0`. Diffusers true CFG performs
separate conditional and unconditional transformer calls per denoising step, which
requires separate cache state for each branch.
