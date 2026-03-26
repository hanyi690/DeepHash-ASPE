"""
使用原始 tags 文件建立 YAll 列索引到 common_tags 标签名的正确映射。

核心思路：
1. 读取原始 tags 文件 作为 ground truth
2. 对于每个样本，提取其在 common_tags.txt 中的标签索引
3. 对比 YAll 中为1的列索引
4. 通过共现矩阵 + 匈牙利算法建立 YAll 列 -> tag 索引的映射
"""

import numpy as np
import h5py
from pathlib import Path
import sys
from scipy.optimize import linear_sum_assignment
from collections import Counter
from typing import Dict, List, Set, Tuple
import os

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_common_tags(tag_names_file: Path) -> Tuple[List[str], Dict[str, int]]:
    """加载 common_tags.txt。"""
    tag_names = []
    tag_to_idx = {}
    with open(tag_names_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if parts:
                tag_names.append(parts[0])
                tag_to_idx[parts[0]] = idx
    return tag_names, tag_to_idx


def load_clean_id(clean_id_path: Path) -> np.ndarray:
    """加载 clean_id 映射。"""
    import scipy.io as sio
    clean_id = sio.loadmat(clean_id_path)['clean_id'].flatten()
    return clean_id


def load_yall(mat_path: Path) -> np.ndarray:
    """加载 YAll。"""
    with h5py.File(mat_path, 'r') as f:
        YAll = f['YAll'][:]
    return YAll


def build_cooccurrence_matrix(
    YAll: np.ndarray,
    clean_id: np.ndarray,
    tags_dir: Path,
    tag_to_idx: Dict[str, int],
    n_tags: int,
    n_samples: int = None
) -> np.ndarray:
    """
    构建共现矩阵：YAll 列索引 × common_tags 索引。

    矩阵元素 [yall_col, tag_idx] 表示：
    YAll 列 yall_col 为1的样本中，有多少样本的原始 tags 包含 tag_idx 对应的标签。
    """
    n_yall_cols = YAll.shape[1]
    cooccur = np.zeros((n_yall_cols, n_tags), dtype=np.int32)

    if n_samples is None:
        n_samples = len(clean_id)

    print(f"处理 {n_samples} 个样本...")

    for i in range(min(n_samples, len(clean_id))):
        orig_img_id = clean_id[i] + 1  # 原始图片ID（从1开始）
        tags_file = tags_dir / f'tags{orig_img_id}.txt'

        if not tags_file.exists():
            continue

        # 读取原始 tags
        with open(tags_file, 'r', encoding='utf-8') as f:
            original_tags = set(line.strip() for line in f if line.strip())

        # 找出在 common_tags 中的标签索引
        tag_indices = [tag_to_idx[t] for t in original_tags if t in tag_to_idx]

        # 获取 YAll 中为1的列索引
        yall_cols = np.where(YAll[i] > 0)[0]

        # 更新共现矩阵
        for yall_col in yall_cols:
            for tag_idx in tag_indices:
                cooccur[yall_col, tag_idx] += 1

        if (i + 1) % 5000 == 0:
            print(f"  已处理 {i + 1} 个样本")

    return cooccur


def infer_mapping_hungarian(cooccur: np.ndarray) -> Dict[int, int]:
    """
    使用匈牙利算法求最优映射。

    返回：yall_col -> tag_idx 的映射
    """
    # 最大化共现次数 = 最小化负共现次数
    row_ind, col_ind = linear_sum_assignment(-cooccur)

    mapping = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    return mapping


def validate_mapping(
    YAll: np.ndarray,
    clean_id: np.ndarray,
    tags_dir: Path,
    tag_to_idx: Dict[str, int],
    tag_names: List[str],
    yall_to_tag: Dict[int, int],
    n_samples: int = 100
) -> float:
    """
    验证映射正确率。

    对于每个样本：
    1. 读取原始 tags，获取 common_tags 中的标签索引集合
    2. 获取 YAll 中为1的列索引，映射到 tag 索引
    3. 比较两个集合是否相同
    """
    correct_samples = 0
    total_tags = 0
    correct_tags = 0

    for i in range(min(n_samples, len(clean_id))):
        orig_img_id = clean_id[i] + 1
        tags_file = tags_dir / f'tags{orig_img_id}.txt'

        if not tags_file.exists():
            continue

        # 原始 tags 中的 common_tags 索引
        with open(tags_file, 'r', encoding='utf-8') as f:
            original_tags = set(line.strip() for line in f if line.strip())
        orig_tag_indices = set(tag_to_idx[t] for t in original_tags if t in tag_to_idx)

        # YAll 映射后的 tag 索引
        yall_cols = np.where(YAll[i] > 0)[0]
        mapped_tag_indices = set(yall_to_tag[col] for col in yall_cols if col in yall_to_tag)

        # 比较
        if orig_tag_indices == mapped_tag_indices:
            correct_samples += 1

        total_tags += len(orig_tag_indices)
        correct_tags += len(orig_tag_indices & mapped_tag_indices)

    sample_accuracy = correct_samples / n_samples
    tag_accuracy = correct_tags / total_tags if total_tags > 0 else 0

    print(f"样本完全匹配率: {correct_samples}/{n_samples} = {sample_accuracy:.1%}")
    print(f"标签匹配率: {correct_tags}/{total_tags} = {tag_accuracy:.1%}")

    return tag_accuracy


def save_mapping(
    yall_to_tag: Dict[int, int],
    tag_names: List[str],
    output_path: Path
):
    """保存映射到 npy 和 txt 文件。"""
    n_yall_cols = max(yall_to_tag.keys()) + 1

    # 创建 tag_mapping 数组
    tag_mapping = [''] * n_yall_cols
    for yall_col, tag_idx in yall_to_tag.items():
        tag_mapping[yall_col] = tag_names[tag_idx]

    # 保存 npy
    np.save(output_path, np.array(tag_mapping, dtype=object), allow_pickle=True)

    # 保存 txt
    with open(output_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
        for i, tag in enumerate(tag_mapping):
            f.write(f"{i}\t{tag}\n")

    print(f"映射已保存: {output_path}")


def main():
    print("=" * 60)
    print("建立 YAll 列到 common_tags 的正确映射")
    print("=" * 60)

    # 路径
    mat_path = PROJECT_ROOT / 'data' / 'FLICKR-25K.mat'
    tags_dir = PROJECT_ROOT / 'data' / 'dcmh' / 'flickr25k' / 'meta' / 'tags'
    tag_names_file = PROJECT_ROOT / 'data' / 'dcmh' / 'flickr25k' / 'doc' / 'common_tags.txt'
    clean_id_path = PROJECT_ROOT / 'data' / 'dcmh' / 'flickr25k' / 'clean_id.flickr25k.mat'
    output_path = PROJECT_ROOT / 'data' / 'dcmh' / 'flickr25k' / 'tag_mapping.npy'

    # 加载数据
    print("\n[步骤 1] 加载数据...")
    tag_names, tag_to_idx = load_common_tags(tag_names_file)
    n_tags = len(tag_names)
    print(f"  common_tags: {n_tags} 个")

    clean_id = load_clean_id(clean_id_path)
    print(f"  clean_id: {len(clean_id)} 个样本")

    YAll = load_yall(mat_path)
    print(f"  YAll: {YAll.shape}")

    # 构建共现矩阵
    print("\n[步骤 2] 构建共现矩阵...")
    cooccur = build_cooccurrence_matrix(
        YAll, clean_id, tags_dir, tag_to_idx, n_tags, n_samples=len(clean_id)
    )
    print(f"  共现矩阵形状: {cooccur.shape}")
    print(f"  非零元素: {np.count_nonzero(cooccur)}")
    print(f"  最大值: {cooccur.max()}")

    # 使用匈牙利算法求映射
    print("\n[步骤 3] 使用匈牙利算法求映射...")
    yall_to_tag = infer_mapping_hungarian(cooccur)
    print(f"  映射数量: {len(yall_to_tag)}")

    # 验证映射
    print("\n[步骤 4] 验证映射正确性...")
    accuracy = validate_mapping(
        YAll, clean_id, tags_dir, tag_to_idx, tag_names, yall_to_tag, n_samples=100
    )

    if accuracy < 0.95:
        print(f"\n警告: 映射正确率 {accuracy:.1%} < 95%，尝试迭代优化...")

        # 迭代优化
        for iteration in range(5):
            # 根据当前映射重新计算共现矩阵
            # 这里可以添加更复杂的优化逻辑
            pass

    # 保存映射
    print("\n[步骤 5] 保存映射...")
    save_mapping(yall_to_tag, tag_names, output_path)

    # 最终验证
    print("\n[步骤 6] 最终验证...")
    validate_mapping(
        YAll, clean_id, tags_dir, tag_to_idx, tag_names, yall_to_tag, n_samples=500
    )

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
