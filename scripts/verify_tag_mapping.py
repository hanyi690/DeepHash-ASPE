"""
验证 YAll 列顺序与 common_tags.txt 行顺序的对应关系。

验证方法：
1. 读取 FLICKR-25K.mat 中的 YAll
2. 选择样本，获取其非零列索引
3. 用 clean_id 映射找到对应的原始图像
4. 与原始 tags 文件对比，验证是否匹配
"""

import numpy as np
import h5py
from pathlib import Path
import sys
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def verify_mapping():
    """验证映射正确性。"""
    # 路径
    mat_path = PROJECT_ROOT / 'data' / 'FLICKR-25K.mat'
    tags_dir = PROJECT_ROOT / 'data' / 'dcmh' / 'flickr25k' / 'meta' / 'tags'
    tag_names_file = PROJECT_ROOT / 'data' / 'dcmh' / 'flickr25k' / 'doc' / 'common_tags.txt'
    clean_id_file = PROJECT_ROOT / 'data' / 'dcmh' / 'flickr25k' / 'clean_id.flickr25k.mat'
    tag_mapping_file = PROJECT_ROOT / 'data' / 'dcmh' / 'flickr25k' / 'tag_mapping.npy'

    print("=" * 60)
    print("验证 YAll 列顺序与 common_tags.txt 行顺序对应关系")
    print("=" * 60)

    # 加载 clean_id 映射
    clean_id = None
    if clean_id_file.exists():
        clean_id = loadmat(clean_id_file)['clean_id'].flatten()
        print(f"\n[0] 加载 clean_id 映射: {len(clean_id)} 个")

    # 读取 tag 名称列表
    tag_names = []
    with open(tag_names_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                tag_names.append(parts[0])

    print(f"[1] common_tags.txt 共 {len(tag_names)} 个标签")

    # 加载 YAll 到 tag 名称的映射（如果存在）
    yall_to_tag = None
    if tag_mapping_file.exists():
        yall_to_tag = list(np.load(tag_mapping_file, allow_pickle=True))
        print(f"[2] 加载 tag_mapping.npy: {len(yall_to_tag)} 个映射")

    # 读取 YAll
    with h5py.File(mat_path, 'r') as f:
        yall = f['YAll'][:]

    print(f"[3] YAll 形状: {yall.shape}")

    # 验证几个样本
    print(f"\n[4] 验证样本（使用 clean_id 映射）:")
    print("-" * 60)

    # 选择几个有足够标签的样本（清洗后索引）
    test_indices = [0, 9, 100, 500, 1000, 2000]

    for cleaned_idx in test_indices:
        # 获取原始图像 ID
        if clean_id is not None:
            original_id = clean_id[cleaned_idx]
        else:
            original_id = cleaned_idx

        # 获取该样本的 YAll 向量
        yall_vec = np.array(yall[cleaned_idx].toarray()).flatten() if hasattr(yall[cleaned_idx], 'toarray') else yall[cleaned_idx]

        # 获取非零列索引
        yall_nonzero = np.where(yall_vec > 0)[0]

        # 获取对应的 tag 名称
        if yall_to_tag is not None:
            yall_tags = [str(yall_to_tag[i]) for i in yall_nonzero if i < len(yall_to_tag) and yall_to_tag[i]]
        else:
            yall_tags = [tag_names[i] for i in yall_nonzero if i < len(tag_names)]

        # 读取原始 tags 文件（使用原始图像 ID）
        tags_file = tags_dir / f'tags{original_id + 1}.txt'
        original_tags = set()
        if tags_file.exists():
            with open(tags_file, 'r', encoding='utf-8') as f:
                for line in f:
                    tag = line.strip()
                    if tag:
                        original_tags.add(tag)

        # 比较
        yall_tags_set = set(yall_tags)
        match = yall_tags_set == original_tags

        print(f"\n样本 cleaned_idx={cleaned_idx} (im{original_id + 1}.jpg):")
        print(f"  YAll 非零索引数: {len(yall_nonzero)}")
        print(f"  原始 tags 文件标签数: {len(original_tags)}")
        print(f"  匹配结果: {'✓ 正确' if match else '✗ 不匹配'}")

        if not match:
            only_in_yall = yall_tags_set - original_tags
            only_in_original = original_tags - yall_tags_set
            if only_in_yall:
                print(f"  仅在 YAll 中: {list(only_in_yall)[:5]}...")
            if only_in_original:
                print(f"  仅在原始文件中: {list(only_in_original)[:5]}...")
        else:
            # 显示部分标签
            print(f"  标签: {yall_tags[:5]}..." if len(yall_tags) > 5 else f"  标签: {yall_tags}")

    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)


if __name__ == '__main__':
    verify_mapping()