## ResilPhase-FLUX

### 1. Set Up Environment

Follow the FLUX repository installation procedure to create the environment:

```bash
conda create -n flux python=3.10
conda activate flux
pip install -e ".[all]"
```

If you prefer `venv`, the original FLUX `README.md` in this repository also works.

### 2. Download Checkpoints with Your Hugging Face Token

If you have trouble connecting to Hugging Face, you can use the mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Then download the required checkpoints:

```bash
huggingface-cli download --token YOUR_HF_TOKEN --resume-download black-forest-labs/FLUX.1-dev --local-dir /path/to/save/pretrained_models/black-forest-labs/FLUX.1-dev
huggingface-cli download --token YOUR_HF_TOKEN --resume-download black-forest-labs/FLUX.1-schnell --local-dir /path/to/save/pretrained_models/black-forest-labs/FLUX.1-schnell
huggingface-cli download --token YOUR_HF_TOKEN --resume-download google/t5-v1_1-xxl --local-dir /path/to/save/pretrained_models/google/t5-v1_1-xxl
huggingface-cli download --token YOUR_HF_TOKEN --resume-download openai/clip-vit-large-patch14 --local-dir /path/to/save/pretrained_models/openai/clip-vit-large-patch14
```

<details>
  <summary>Download Checkpoints on AutoDL</summary>

```bash
huggingface-cli download --token YOUR_HF_TOKEN --resume-download black-forest-labs/FLUX.1-dev --local-dir /root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev
huggingface-cli download --token YOUR_HF_TOKEN --resume-download black-forest-labs/FLUX.1-schnell --local-dir /root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-schnell
huggingface-cli download --token YOUR_HF_TOKEN --resume-download google/t5-v1_1-xxl --local-dir /root/autodl-tmp/pretrained_models/google/t5-v1_1-xxl
huggingface-cli download --token YOUR_HF_TOKEN --resume-download openai/clip-vit-large-patch14 --local-dir /root/autodl-tmp/pretrained_models/openai/clip-vit-large-patch14
```
</details>

### 3. Set Environment Variables

Set the FLUX checkpoint paths in your shell environment:

```bash
export FLUX_SCHNELL="/path/to/save/pretrained_models/black-forest-labs/FLUX.1-schnell/flux1-schnell.safetensors"
export FLUX_DEV="/path/to/save/pretrained_models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors"
export AE="/path/to/save/pretrained_models/black-forest-labs/FLUX.1-dev/ae.safetensors"
```

<details>
  <summary>Set Environment Variables for AutoDL</summary>

```bash
export FLUX_SCHNELL="/root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-schnell/flux1-schnell.safetensors"
export FLUX_DEV="/root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors"
export AE="/root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev/ae.safetensors"
```
</details>

### 4. Sampling with ResilPhase-FLUX

#### Interactive Sampling

```bash
python -m flux --name <name> --loop
```

#### Single Sample Generation

```bash
python -m flux --name <name> \
  --height <height> --width <width> \
  --prompt "<prompt>"
```

Typically, `<name>` should be set to `flux-dev`.

#### Batch Sampling with Prompt File

```bash
python src/sample.py --prompt_file </path/to/your/prompt.txt> \
  --width 1024 --height 1024 --model_name flux-dev \
  --add_sampling_metadata --output_dir </path/to/your/generated/samples/folder> --num_steps 50
```

The `--add_sampling_metadata` parameter determines whether the prompt is embedded in the image EXIF metadata.

#### FLOPs Testing

```bash
python src/sample.py --prompt_file </path/to/your/test/prompt.txt> \
  --width 1024 --height 1024 --model_name flux-dev \
  --add_sampling_metadata --output_dir </path/to/your/generated/samples/folder> \
  --num_steps 50 --test_FLOPs
```

When `--test_FLOPs` is enabled, the code measures the inference cost without normal image generation output.

> **Note:** For FLOPs testing, make sure `src/flux/math.py` uses the naive attention path instead of FlashAttention / `torch.nn.functional.scaled_dot_product_attention`.

### 5. ResilPhase Configuration

The ResilPhase method configuration is defined directly in:

`ResilPhase-FLUX/src/flux/modules/cache_functions/cache_init.py`

The current implementation uses the `ResilPhase` mode in that file. You can adjust:

- `fresh_threshold`: controls how often full computation is performed.
- `max_order`: controls the interpolation order used by ResilPhase.
- `mapping_method`: phase-axis mapping method, choose from `balanced` or `chebyshev`.
- `balance_alpha`: hyperparameter for balanced mapping, only used when `mapping_method == 'balanced'`.

At the moment, the default ResilPhase configuration in `cache_init.py` is:

```python
mode = 'ResilPhase'
cache_dic['fresh_threshold'] = 6
cache_dic['max_order'] = 1
cache_dic['mapping_method'] = 'balanced'
cache_dic['balance_alpha'] = 0.55
```

If you want to switch to Chebyshev-node mapping, change:

```python
cache_dic['mapping_method'] = 'chebyshev'
```

The corresponding phase mapping and interpolation logic is implemented in:

`ResilPhase-FLUX/src/flux/resilphase_utils/__init__.py`

### 6. Notes

- `src/sample.py` already runs through the ResilPhase cache denoising path.
- As long as `mode = 'ResilPhase'` in `cache_init.py`, the sampling code will use the ResilPhase phase-mapping mechanism automatically.
- This repository keeps the original FLUX files while adding the ResilPhase acceleration logic on top of them.
