#!/usr/bin/env python3
"""
统计指定文件夹下的文件数量
"""
import os
import sys
from pathlib import Path


def count_files(directory, recursive=False, pattern=None):
    """
    统计文件夹下的文件数量
    
    Args:
        directory: 文件夹路径
        recursive: 是否递归统计子文件夹
        pattern: 文件匹配模式（如 "*.mp4"）
    
    Returns:
        文件数量
    """
    path = Path(directory)
    
    if not path.exists():
        print(f"错误：路径不存在: {directory}")
        return 0
    
    if not path.is_dir():
        print(f"错误：不是文件夹: {directory}")
        return 0
    
    if recursive:
        if pattern:
            files = list(path.rglob(pattern))
        else:
            files = [f for f in path.rglob("*") if f.is_file()]
    else:
        if pattern:
            files = list(path.glob(pattern))
        else:
            files = [f for f in path.iterdir() if f.is_file()]
    
    return len(files)


def main():
    if len(sys.argv) < 2:
        print("用法：")
        print("  python count_files.py <文件夹路径>")
        print("  python count_files.py <文件夹路径> -r           # 递归统计")
        print("  python count_files.py <文件夹路径> -p '*.mp4'   # 指定文件类型")
        print("  python count_files.py <文件夹路径> -r -p '*.mp4'")
        print()
        print("示例：")
        print("  python count_files.py ./samples")
        print("  python count_files.py ./samples -r")
        print("  python count_files.py ./samples -p '*.mp4'")
        sys.exit(1)
    
    directory = sys.argv[1]
    recursive = '-r' in sys.argv or '--recursive' in sys.argv
    
    # 查找 pattern
    pattern = None
    if '-p' in sys.argv:
        idx = sys.argv.index('-p')
        if idx + 1 < len(sys.argv):
            pattern = sys.argv[idx + 1]
    elif '--pattern' in sys.argv:
        idx = sys.argv.index('--pattern')
        if idx + 1 < len(sys.argv):
            pattern = sys.argv[idx + 1]
    
    count = count_files(directory, recursive, pattern)
    
    print(f"文件夹: {directory}")
    if pattern:
        print(f"匹配模式: {pattern}")
    if recursive:
        print(f"递归统计: 是")
    print(f"文件数量: {count}")


if __name__ == "__main__":
    main()

