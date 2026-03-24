"""
修复检查点文件

将旧格式检查点重新保存为新格式，解决加载卡住问题。

使用方法：
    python scripts/fix_checkpoint.py <checkpoint_path> [--output <output_path>]

示例：
    # 覆盖原文件
    python scripts/fix_checkpoint.py results/flickr-25k/20260323_183559/checkpoint.pth

    # 保存到新文件
    python scripts/fix_checkpoint.py results/flickr-25k/20260323_183559/checkpoint.pth --output results/flickr-25k/20260323_183559/checkpoint_fixed.pth
"""

import sys
import os
import argparse
import zipfile

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def validate_checkpoint(checkpoint_path):
    """
    验证检查点文件完整性。

    参数：
        checkpoint_path: 检查点文件路径

    返回：
        (bool, str): (是否有效, 错误信息)
    """
    try:
        with zipfile.ZipFile(checkpoint_path, 'r') as z:
            # 检查必要文件存在
            required = ['checkpoint/data.pkl', 'checkpoint/version']
            for f in required:
                if f not in z.namelist():
                    return False, f"缺少文件: {f}"
        return True, "检查点文件完整"
    except zipfile.BadZipFile:
        return False, "检查点文件损坏（无效的 ZIP 格式）"
    except Exception as e:
        return False, str(e)


def fix_checkpoint(input_path, output_path=None):
    """
    修复检查点文件。

    将旧格式检查点重新保存为新格式，确保张量存储正确序列化。

    参数：
        input_path: 输入检查点路径
        output_path: 输出检查点路径（默认覆盖原文件）
    """
    if output_path is None:
        output_path = input_path

    # 验证输入文件
    is_valid, msg = validate_checkpoint(input_path)
    if not is_valid:
        print(f"错误: {msg}")
        return False

    print(f"检查点验证通过: {msg}")

    # 获取文件大小
    file_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f"检查点大小: {file_size:.2f} MB")

    # 加载检查点（强制使用 CPU）
    print(f"正在加载检查点: {input_path}")
    print("注意: 使用 CPU 加载，可能需要较长时间...")

    try:
        checkpoint = torch.load(
            input_path,
            map_location='cpu',
            weights_only=False
        )
    except Exception as e:
        print(f"加载失败: {e}")
        return False

    print("加载成功!")

    # 显示检查点内容
    print("\n检查点内容:")
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor {value.shape}, dtype={value.dtype}, device={value.device}")
        elif isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} keys")
        elif isinstance(value, (int, float, str)):
            print(f"  {key}: {type(value).__name__} = {value}")
        else:
            print(f"  {key}: {type(value).__name__}")

    # 确保所有张量在 CPU 上
    for key in ['F_buffer', 'G_buffer', 'B']:
        if key in checkpoint and isinstance(checkpoint[key], torch.Tensor):
            checkpoint[key] = checkpoint[key].cpu()

    # 重新保存（使用新的 ZIP 序列化格式）
    print(f"\n正在重新保存检查点: {output_path}")
    torch.save(checkpoint, output_path, _use_new_zipfile_serialization=True)

    # 验证输出文件
    is_valid, msg = validate_checkpoint(output_path)
    if is_valid:
        new_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"修复完成! 新文件大小: {new_size:.2f} MB")
        return True
    else:
        print(f"修复后验证失败: {msg}")
        return False


def main():
    parser = argparse.ArgumentParser(description='修复检查点文件')
    parser.add_argument('checkpoint', help='检查点文件路径')
    parser.add_argument('--output', '-o', help='输出文件路径（默认覆盖原文件）')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"错误: 文件不存在 {args.checkpoint}")
        return

    success = fix_checkpoint(args.checkpoint, args.output)
    if success:
        print("\n修复成功!")
    else:
        print("\n修复失败!")
        sys.exit(1)


if __name__ == '__main__':
    main()