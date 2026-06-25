#!/usr/bin/env python3
import os
# First, set environment variables to disable tokenizers warning messages
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import json
import csv
import contextlib
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

@contextlib.contextmanager
def suppress_output():
    """
    Temporarily suppress standard output, standard error, and loguru output.
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            # Disable loguru output
            logger.disable("")
            yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logger.enable("")

def load_prompts_from_file(file_path, prompt_column=None):
    """
    从不同格式的文件中加载提示词
    支持的格式：JSON, TXT, CSV
    
    Args:
        file_path: 文件路径
        prompt_column: CSV文件中提示词所在的列名或索引（默认为第一列）
    
    Returns:
        list: 包含 {"prompt_en": prompt} 格式的字典列表
    """
    file_ext = Path(file_path).suffix.lower()
    prompts = []
    
    if file_ext == '.json':
        # JSON 格式
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'prompt_en' in item:
                    prompts.append(item)
                elif isinstance(item, str):
                    prompts.append({"prompt_en": item})
        elif isinstance(data, dict) and 'prompt_en' in data:
            prompts.append(data)
            
    elif file_ext == '.txt':
        # TXT 格式：每行一个提示词
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 忽略空行
                    prompts.append({"prompt_en": line})
                    
    elif file_ext == '.csv':
        # CSV 格式
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader, None)  # 读取标题行
            
            # 确定提示词列的索引
            prompt_idx = 0  # 默认第一列
            if prompt_column is not None:
                if isinstance(prompt_column, str) and headers:
                    # 按列名查找
                    if prompt_column in headers:
                        prompt_idx = headers.index(prompt_column)
                    else:
                        logger.warning(f"Column '{prompt_column}' not found in CSV headers: {headers}")
                elif isinstance(prompt_column, int):
                    # 按列索引
                    prompt_idx = prompt_column
            
            for row in reader:
                if len(row) > prompt_idx and row[prompt_idx].strip():
                    prompts.append({"prompt_en": row[prompt_idx].strip()})
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .json, .txt, .csv")
    
    logger.info(f"Loaded {len(prompts)} prompts from {file_path}")
    return prompts

def main():
    # Get command-line arguments
    args = parse_args()

    # New parameter: If --vbench-json-path is provided, load the file; otherwise, keep the original logic (single prompt)
    # Now supports JSON, TXT, and CSV formats
    vbench_json_path = getattr(args, "vbench_json_path", None)  # 重命名为更通用的参数名
    prompt_file_path = getattr(args, "prompt_file_path", vbench_json_path)  # 支持新的参数名
    prompt_column = getattr(args, "prompt_column", None)  # CSV文件中提示词列名或索引
    index_start = int(getattr(args, "index_start", 0))
    index_end   = int(getattr(args, "index_end", -1))
    num_videos_per_prompt = int(getattr(args, "num_videos_per_prompt", 1))
    
    if prompt_file_path:
        if not os.path.isfile(prompt_file_path):
            raise ValueError(f"Prompt file not found: {prompt_file_path}")
        
        prompts_data = load_prompts_from_file(prompt_file_path, prompt_column)
        
        if index_end < 0 or index_end >= len(prompts_data):
            index_end = len(prompts_data) - 1
        selected_prompts = prompts_data[index_start:index_end+1]
        logger.info(f"Processing prompts {index_start} to {index_end} from file: {prompt_file_path}")
    else:
        # Original logic: Single prompt only
        selected_prompts = [{"prompt_en": args.prompt}]
    
    # Load model
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` does not exist: {models_root_path}")
    
    # Create save directory
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    os.makedirs(save_path, exist_ok=True)
    
    # Load the sampler (only load the model once)
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    # Update sampler internal parameters
    args = hunyuan_video_sampler.args

    # Initialize timing variables
    all_inference_times = []
    all_diffusion_times = []
    total_videos_generated = 0

    total_prompts = len(selected_prompts)
    for idx, item in enumerate(selected_prompts):
        prompt_text = item.get("prompt_en", "")
        logger.info(f"Starting inference for Prompt [{idx+1}/{total_prompts}]: {prompt_text}")
        for seed_offset in range(num_videos_per_prompt):
            # Calculate video_counter based on index_start and current position
            # This ensures consistent numbering even when resuming from middle
            video_counter = (index_start + idx) * num_videos_per_prompt + seed_offset + 1
            current_seed = args.seed + seed_offset
            cur_save_path = f"{save_path}/{video_counter}_{prompt_text}-{seed_offset}.mp4"

            # Check if the target file already exists
            if os.path.exists(cur_save_path):
                logger.info(f"Video already exists, skipping: {cur_save_path}")
                continue  # Skip this video and proceed to the next one

            # Start timing for this video
            video_start_time = time.time()
            
            with suppress_output():
                outputs = hunyuan_video_sampler.predict(
                    prompt=prompt_text,
                    height=args.video_size[0],
                    width=args.video_size[1],
                    video_length=args.video_length,
                    seed=current_seed,
                    negative_prompt=args.neg_prompt,
                    infer_steps=args.infer_steps,
                    guidance_scale=args.cfg_scale,
                    num_videos_per_prompt=1,
                    flow_shift=args.flow_shift,
                    batch_size=args.batch_size,
                    embedded_guidance_scale=args.embedded_cfg_scale,
                )
            
            # End timing for this video
            video_end_time = time.time()
            video_inference_time = video_end_time - video_start_time
            all_inference_times.append(video_inference_time)
            
            # Extract diffusion sampling time from the outputs if available
            diffusion_time = outputs.get('sampling_time', 0.0)
            if diffusion_time > 0:
                all_diffusion_times.append(diffusion_time)
            
            total_videos_generated += 1
            
            # Log the inference time for this video
            logger.info(f"Video inference completed in {video_inference_time:.2f} seconds for: {prompt_text}-{seed_offset}")
            if diffusion_time > 0:
                logger.info(f"Pure diffusion sampling time for this video: {diffusion_time:.4f} seconds")
            
            samples = outputs['samples']
            for i, sample in enumerate(samples):
                sample = samples[i].unsqueeze(0)
                save_videos_grid(sample, cur_save_path, fps=24)
                logger.info(f"Sample saved to: {cur_save_path}")

    # Calculate and print overall statistics
    if all_inference_times:
        average_inference_time = sum(all_inference_times) / len(all_inference_times)
        total_inference_time = sum(all_inference_times)
        
        logger.info("=" * 60)
        logger.info("📊 INFERENCE TIME STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total videos generated: {total_videos_generated}")
        logger.info(f"Total inference time: {total_inference_time:.2f} seconds")
        logger.info(f"Average inference time per video: {average_inference_time:.2f} seconds")
        logger.info(f"Min inference time: {min(all_inference_times):.2f} seconds")
        logger.info(f"Max inference time: {max(all_inference_times):.2f} seconds")
        
        # Display diffusion sampling statistics if available
        if all_diffusion_times:
            average_diffusion_time = sum(all_diffusion_times) / len(all_diffusion_times)
            total_diffusion_time = sum(all_diffusion_times)
            logger.info("-" * 60)
            logger.info("🔥 PURE DIFFUSION SAMPLING TIME STATISTICS")
            logger.info("-" * 60)
            logger.info(f"Total pure diffusion sampling time: {total_diffusion_time:.4f} seconds")
            logger.info(f"Average pure diffusion sampling time per video: {average_diffusion_time:.4f} seconds")
            logger.info(f"Min pure diffusion sampling time: {min(all_diffusion_times):.4f} seconds")
            logger.info(f"Max pure diffusion sampling time: {max(all_diffusion_times):.4f} seconds")
        else:
            logger.info("No pure diffusion sampling time data collected.")
            
        logger.info("=" * 60)
    else:
        logger.info("No new videos were generated (all videos already existed).")

    
if __name__ == "__main__":
    main()
