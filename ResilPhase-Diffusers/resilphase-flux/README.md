# Resilphase-FLUX-Diffusers

`Resilphase-FLUX-Diffusers` adapts the ResilPhase-FLUX acceleration path to
HuggingFace Diffusers FLUX pipelines.

It does not modify the installed `diffusers` package. Instead, it patches a loaded
`FluxPipeline` or `FluxTransformer2DModel` instance at runtime.

## Install

```bash
cd /mnt/public/zqc/ResilPhase/ResilPhase-Diffusers/resilphase-flux
pip install -e .
```

The distribution/package name is `Resilphase-FLUX-Diffusers`. The Python import
module is `resilphase_diffusers`.

## Usage

```python
import torch
from diffusers import FluxPipeline
from resilphase_diffusers import apply_resilphase

num_steps = 50

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
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

image = pipe(
    "An image of a squirrel in Picasso style",
    num_inference_steps=num_steps,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("resilphase_flux_diffusers.png")
```

## Configuration

- `num_steps`: must match `num_inference_steps` for the current pipeline call.
- `fresh_threshold`: interval for full transformer computation.
- `first_enhance`: number of initial full-computation steps.
- `max_order`: interpolation order. The cache keeps `max_order + 1` full-computation samples.
- `mapping_method`: `balanced` or `chebyshev`.
- `balance_alpha`: phase-axis compression strength for `balanced` mapping.

## Current Scope

This ResilPhase-FLUX Diffusers patch targets `diffusers>=0.35.0` and the current
`FluxTransformer2DModel.forward` shape where single-stream blocks return
`(encoder_hidden_states, hidden_states)`.

ControlNet inputs are currently routed through the original Diffusers transformer
forward as a correctness fallback, so ResilPhase acceleration is disabled for those calls.

The implementation follows the ResilPhase-FLUX design:

- full steps run all double-stream and single-stream transformer blocks;
- full steps cache whole-stack deltas for image/text double blocks and concatenated single blocks;
- cache steps predict those deltas with phase-axis barycentric Lagrange interpolation.
