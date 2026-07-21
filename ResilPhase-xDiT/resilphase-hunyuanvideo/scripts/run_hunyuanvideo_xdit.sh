set -x

export PYTHONPATH=$PWD/src:$PYTHONPATH

MODEL_ID=${MODEL_ID:-hunyuanvideo-community/HunyuanVideo}
N_GPUS=${N_GPUS:-1}
PROMPT=${PROMPT:-A cat walks on the grass, realistic style.}
INFERENCE_STEP=${INFERENCE_STEP:-50}
OUTPUT_DIR=${OUTPUT_DIR:-./results}

# The current xFuser release in this environment has no HunyuanVideoTransformer3D
# tensor/sequence/pipeline wrapper. Keep those degrees at 1; data parallelism is
# supported for prompt lists.
PARALLEL_ARGS="--data_parallel_degree $N_GPUS --pipefusion_parallel_degree 1 --ulysses_degree 1 --ring_degree 1 --tensor_parallel_degree 1"

torchrun --nproc_per_node=$N_GPUS examples/text_to_video_xdit.py \
  --model "$MODEL_ID" \
  $PARALLEL_ARGS \
  --height 480 \
  --width 640 \
  --num_frames 65 \
  --num_inference_steps "$INFERENCE_STEP" \
  --warmup_steps 1 \
  --prompt "$PROMPT" \
  --guidance_scale 6.0 \
  --true-cfg-scale 1.0 \
  --output-dir "$OUTPUT_DIR" \
  --fresh-threshold 6 \
  --max-order 1 \
  --first-enhance 3 \
  --mapping-method balanced \
  --dtype bfloat16
