set -x

export PYTHONPATH=$PWD/src:$PYTHONPATH

MODEL_ID=${MODEL_ID:-/root/autodl-tmp/black-forest-labs/FLUX.1-dev}
N_GPUS=${N_GPUS:-8}
PROMPT=${PROMPT:-A dog standing on the grass.}
INFERENCE_STEP=${INFERENCE_STEP:-50}
OUTPUT_DIR=${OUTPUT_DIR:-./results}

TASK_ARGS="--height 1024 --width 1024 --no_use_resolution_binning"
PARALLEL_ARGS="--pipefusion_parallel_degree 1 --ulysses_degree $N_GPUS --ring_degree 1 --tensor_parallel_degree 1"

torchrun --nproc_per_node=$N_GPUS examples/flux_text_to_image_xdit.py \
  --model "$MODEL_ID" \
  $PARALLEL_ARGS \
  $TASK_ARGS \
  --num_inference_steps "$INFERENCE_STEP" \
  --warmup_steps 1 \
  --prompt "$PROMPT" \
  --guidance_scale 0.0 \
  --output-dir "$OUTPUT_DIR" \
  --fresh-threshold 6 \
  --max-order 1 \
  --first-enhance 3 \
  --mapping-method balanced \
  --dtype bfloat16
