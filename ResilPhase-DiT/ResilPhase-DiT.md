## ResilPhase-DiT

### 1. Prepare Environment

```bash
cd ResilPhase-DiT
conda env create -f environment.yml
conda activate DiT
pip install flash-attention
```

### 2. Download Checkpoints

Simply follow the official DiT documentation to download the necessary checkpoints.

In the current codebase, the default checkpoint path used in sampling is:

```bash
/mnt/public/zqc/DiT-XL-2-256x256.pt
```

You can also override it with:

```bash
--ckpt /path/to/your/checkpoint.pt
```

### 3. Run Samples

#### Single-Batch Inference

Set your desired class ID in `ResilPhase-DiT/sample.py`.

<details>
  <summary>Recommended Parameter Settings</summary>

  The current implementation supports both phase-axis mappings:
  - `--mapping-method chebyshev`
  - `--mapping-method balanced`

  For the current ResilPhase-DiT code, a practical starting point is:
  - `interval=4`
  - `max_order=4`
  - `mapping_method=chebyshev`

  If you want to test balanced mapping, you can additionally set:
  - `--mapping-method balanced`
  - `--balance-alpha 0.55`

  You can also experiment with different parameter combinations depending on your quality / acceleration tradeoff.
</details>

Run inference with a typical setting:

```bash
python sample.py \
  --ddim-sample \
  --num-sampling-steps 50 \
  --interval 4 \
  --max-order 4 \
  --mapping-method chebyshev
```

Example with balanced mapping:

```bash
python sample.py \
  --ddim-sample \
  --num-sampling-steps 50 \
  --interval 4 \
  --max-order 4 \
  --mapping-method balanced \
  --balance-alpha 0.55
```

#### Distributed Data Parallel (DDP) Inference

```bash
torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py \
  --model DiT-XL/2 \
  --per-proc-batch-size 50 \
  --image-size 256 \
  --cfg-scale 1.5 \
  --ddim-sample \
  --num-sampling-steps 50 \
  --interval 4 \
  --max-order 4 \
  --mapping-method chebyshev \
  --num-fid-samples 50000
```

### 4. ResilPhase Configuration

The ResilPhase interpolation logic is implemented in:

`ResilPhase-DiT/resilphase_utils/__init__.py`

The sampling-time cache configuration is initialized in:

`ResilPhase-DiT/cache_functions/cache_init.py`

The main hyperparameters are:

- `interval`: controls how often full computation is performed.
- `max_order`: controls the interpolation order.
- `mapping_method`: choose from `chebyshev` or `balanced`.
- `balance_alpha`: only used when `mapping_method == 'balanced'`.

In the current code, `mapping_method` and `balance_alpha` are passed from:

- `sample.py`
- `sample_ddp.py`

and then written into `cache_dic` by:

`ResilPhase-DiT/cache_functions/cache_init.py`

### 5. Notes

- `sample.py` and `sample_ddp.py` both support ResilPhase phase mapping.
- `chebyshev` and `balanced` mappings are both implemented in `resilphase_utils/__init__.py`.
- If you want to compare mappings, keep other parameters fixed and change only:
  - `--mapping-method`
  - `--balance-alpha`
