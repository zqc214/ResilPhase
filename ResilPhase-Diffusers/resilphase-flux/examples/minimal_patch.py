import torch
from diffusers import FluxPipeline

from resilphase_diffusers import apply_resilphase


num_steps = 50

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

apply_resilphase(pipe, num_steps=num_steps)

image = pipe(
    "An image of a squirrel in Picasso style",
    num_inference_steps=num_steps,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("resilphase_flux_diffusers.png")
