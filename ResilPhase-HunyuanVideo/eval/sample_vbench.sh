#!/bin/bash
# Usage:
#   ./sample_vbench.sh <full_info_path> <Num_Devices> <SEED> <Num_Samples> <Video_Save_Path> <Path2Log>
# Example:
#   ./eval/sample_vbench.sh ./eval/ 1 42 5 /root/autodl-tmp/HunyuanVideo_samples/vbench-simulate /root/autodl-tmp/HunyuanVideo_vbench_logs/

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <full_info_path> <Num_Devices> <SEED> <Num_Samples> <Video_Save_Path> <Path2Log>"
    exit 1
fi

full_info_path="$1"
Num_Devices="$2"
SEED="$3"
Num_Samples="$4"
Video_Save_Path="$5"
Path2Log="$6"

JSON_FILE="${full_info_path}/VBench_full_info.json"
if [ ! -f "$JSON_FILE" ]; then
    echo "Error: JSON file not found at ${JSON_FILE}"
    exit 1
fi

# Calculate the total number of prompts in the JSON file
if command -v jq &>/dev/null; then
    total_prompts=$(jq '. | length' "$JSON_FILE")
else
    echo "Warning: jq not found, using python for JSON parsing."
    total_prompts=$(python3 -c "import json; f=open('$JSON_FILE'); data=json.load(f); print(len(data))")
fi

echo "Total number of prompts: $total_prompts"

# Compute the number of prompts each device will handle (the last device may have fewer)
chunk_size=$(( (total_prompts + Num_Devices - 1) / Num_Devices ))

# Ensure the log directory exists
mkdir -p "$Path2Log"

# Launch separate background processes for each GPU
for (( d=0; d<Num_Devices; d++ )); do
    {
        index_start=$(( d * chunk_size ))
        index_end=$(( (d+1) * chunk_size - 1 ))
        if [ $index_end -ge $total_prompts ]; then
            index_end=$(( total_prompts - 1 ))
        fi
        
        log_file="${Path2Log}/device_${d}.log"
        echo "Device $d: Processing prompts index range [$index_start, $index_end]" > "$log_file"
        
        CUDA_VISIBLE_DEVICES="$d" python3 sample_video_vbench.py \
            --vbench-json-path "$JSON_FILE" \
            --index-start "$index_start" \
            --index-end "$index_end" \
            --seed "$SEED" \
            --num-videos-per-prompt "$Num_Samples" \
            --video-size 480 640 \
            --video-length 65 \
            --infer-steps 50 \
            --mapping-method balanced \
            --balance-alpha 0.55 \
            --flow-reverse \
            --use-cpu-offload \
            --save-path "$Video_Save_Path" >> "$log_file" 2>&1
        
        echo "Device $d: Completed inference for index range [$index_start, $index_end]" >> "$log_file"
    } &
done

wait
echo "All device tasks have been completed!"
