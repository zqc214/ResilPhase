# Resilphase-DiT-Diffusers

`Resilphase-DiT-Diffusers` adapts the ResilPhase-DiT acceleration path to
HuggingFace Diffusers DiT pipelines.

It patches a loaded `DiTPipeline` or `DiTTransformer2DModel` instance at runtime.
It does not modify the installed `diffusers` package.

## Install

```bash
cd /mnt/public/zqc/ResilPhase/ResilPhase-Diffusers/resilphase-dit
pip install -e .
```

The distribution/package name is `Resilphase-DiT-Diffusers`. The Python import
module is `resilphase_dit_diffusers`.

## Usage

```python
import torch
from diffusers import DiTPipeline
from resilphase_dit_diffusers import apply_resilphase

num_steps = 50

pipe = DiTPipeline.from_pretrained(
    "facebook/DiT-XL-2-256",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

apply_resilphase(
    pipe,
    num_steps=num_steps,
    interval=4,
    max_order=4,
    mapping_method="chebyshev",
)

image = pipe(
    class_labels=[495],
    num_inference_steps=num_steps,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
image.save("resilphase_dit_diffusers.png")
```

## Configuration

- `num_steps`: must match `num_inference_steps` for the current pipeline call.
- `interval`: interval for full transformer-block computation.
- `first_enhance`: number of initial full-computation calls.
- `max_order`: interpolation order. The cache keeps `max_order + 1` full-computation samples.
- `mapping_method`: `chebyshev` or `balanced`.
- `balance_alpha`: phase-axis compression strength for `balanced` mapping.

## Current Scope

This patch targets `diffusers>=0.35.0` and `DiTTransformer2DModel`.

The implementation follows the ResilPhase-DiT design:

- full calls run all transformer blocks;
- full calls cache the whole block-stack delta;
- cache calls skip all transformer blocks and predict that delta with phase-axis
  barycentric Lagrange interpolation.
