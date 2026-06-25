## **ResilPhase-HunyuanVideo**

### **1. Prepare Environment**

Follow the official **HunyuanVideo** documentation to set up the environment.

<details>
  <summary><strong>Conda Environment Setup</strong></summary>

  ```bash
  # 1. Create the Conda environment
  conda create -n HunyuanVideo python==3.10.9

  # 2. Activate the environment
  conda activate HunyuanVideo

  # 3. Install PyTorch and dependencies
  # For CUDA 11.8
  conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia
  # For CUDA 12.4
  conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

  # 4. Install required Python dependencies
  python -m pip install -r requirements.txt

  # 5. Install FlashAttention v2
  python -m pip install ninja
  python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3

  # 6. Install xDiT for parallel inference
  python -m pip install xfuser==0.4.0
  ```

  If you encounter a floating point exception on specific GPUs, refer to the troubleshooting instructions in the repository `README.md`.

</details>

### **2. Download Checkpoints**

Refer to the checkpoint and model preparation instructions in:

`ResilPhase-HunyuanVideo/README.md`

### **3. Run ResilPhase-HunyuanVideo Samples**

#### **Single Video Inference**

Run inference on a single video. Adjust the prompt and output path as needed.

```bash
cd ResilPhase-HunyuanVideo
python3 sample_video.py \
  --video-size 480 640 \
  --video-length 65 \
  --infer-steps 50 \
  --seed 42 \
  --prompt "A cat walks on the grass, realistic style." \
  --flow-reverse \
  --use-cpu-offload \
  --save-path /path/to/save/videos
```

You can also use the provided script:

```bash
./scripts/run_sample_video.sh
```

---

#### **Multi-Video Inference (VBench Testing)**

We provide a script for VBench-style evaluation with multi-GPU parallel inference:

```bash
cd ResilPhase-HunyuanVideo
./eval/sample_vbench.sh ./eval/ 1 42 5 /path/to/save/vbench/videos /path/to/save/logger/files
```

Or run the python entry directly:

```bash
python3 sample_video_vbench.py \
  --vbench-json-path ./eval/VBench_full_info.json \
  --index-start 0 \
  --index-end 9 \
  --seed 42 \
  --num-videos-per-prompt 1 \
  --video-size 480 640 \
  --video-length 65 \
  --infer-steps 50 \
  --flow-reverse \
  --use-cpu-offload \
  --save-path /path/to/save/videos
```

---

<details>
  <summary><strong>Hyperparameter Tuning & Recommendations</strong></summary>

  ResilPhase-HunyuanVideo is currently configured through:

  ```
  ResilPhase-HunyuanVideo/hyvideo/modules/cache_functions/cache_init.py
  ```

  The active method is controlled by the `mode` variable in that file.
  At the moment, the code is set to:

  ```python
  mode = 'ResilPhase'
  ```

  In the `ResilPhase` block, the key hyperparameters are:

  - `fresh_threshold`: controls how often full computation is performed.
  - `max_order`: controls the interpolation order.
  - `mapping_method`: choose from `balanced` or `chebyshev`.
  - `balance_alpha`: only used when `mapping_method == 'balanced'`.

  The current default ResilPhase configuration is:

  ```python
  cache_dic['fresh_threshold'] = 6
  cache_dic['max_order'] = 1
  cache_dic['mapping_method'] = 'balanced'
  cache_dic['balance_alpha'] = 0.55
  ```

  If you want to switch to Chebyshev mapping, change:

  ```python
  cache_dic['mapping_method'] = 'chebyshev'
  ```

</details>

---

<details>
  <summary><strong>About Phase Mapping</strong></summary>

  The corresponding phase-mapping logic is implemented in:

  ```
  ResilPhase-HunyuanVideo/hyvideo/modules/resilphase_utils/__init__.py
  ```

  Both mappings are implemented there:

  - balanced mapping
  - chebyshev-node mapping

  During sampling, `sample_video.py` and `sample_video_vbench.py` both enter the ResilPhase cache path automatically, as long as `mode = 'ResilPhase'` in `cache_init.py`.

</details>

---

<details>
  <summary><strong>About Generation Quality</strong></summary>

  Like other acceleration methods, ResilPhase does not guarantee identical outputs compared to the full non-accelerated baseline.
  You may need to tune `fresh_threshold`, `max_order`, and the mapping strategy for your own quality / speed tradeoff.

</details>
