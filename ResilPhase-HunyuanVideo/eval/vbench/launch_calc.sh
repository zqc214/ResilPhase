# !/bin/bash

VIDEO_DIR=$1
CKPT_DIR=$2
LOG_BASE=$CKPT_DIR
mkdir -p $LOG_BASE
echo "Logging to $LOG_BASE"

# 4GPU配置示例
GPUS=(0 1 2 3)
START_INDEX_LIST=(0 4 8 12)
END_INDEX_LIST=(4 8 12 16)
TASK_ID_LIST=(calc_vbench_1 calc_vbench_2 calc_vbench_3 calc_vbench_4) # for log records only

for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[i]} python eval/vbench/calc_vbench.py $VIDEO_DIR $CKPT_DIR --start ${START_INDEX_LIST[i]} --end ${END_INDEX_LIST[i]} > ${LOG_BASE}/${TASK_ID_LIST[i]}.log 2>&1 &
done
