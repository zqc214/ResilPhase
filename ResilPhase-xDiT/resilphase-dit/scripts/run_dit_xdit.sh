set -x

export PYTHONPATH=$PWD/src:$PYTHONPATH

MODEL_ID=${MODEL_ID:-facebook/DiT-XL-2-256}
N_GPUS=${N_GPUS:-1}
CLASS_LABELS=${CLASS_LABELS:-495}
INFERENCE_STEP=${INFERENCE_STEP:-50}
OUTPUT_DIR=${OUTPUT_DIR:-./results}

# The current xFuser release in this environment has no DiTTransformer2DModel
# tensor/sequence/pipeline wrapper. Keep those degrees at 1; data parallelism is
# supported for multiple class labels.
PARALLEL_ARGS="--data_parallel_degree $N_GPUS --pipefusion_parallel_degree 1 --ulysses_degree 1 --ring_degree 1 --tensor_parallel_degree 1"

torchrun --nproc_per_node=$N_GPUS examples/dit_class_conditional_xdit.py \
  --model "$MODEL_ID" \
  --class-labels "$CLASS_LABELS" \
  --num_inference_steps "$INFERENCE_STEP" \
  --guidance_scale 4.0 \
  --output-dir "$OUTPUT_DIR" \
  --interval 4 \
  --max-order 4 \
  --first-enhance 2 \
  --mapping-method chebyshev \
  --dtype float16 \
  $PARALLEL_ARGS
