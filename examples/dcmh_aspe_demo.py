"""
DCMH + ASPE 集成示例

展示如何在保持 mAP 不变的前提下实现隐私保护检索。

运行此脚本将：
1. 生成模拟的 DCMH 哈希码
2. 计算原始 mAP（明文）
3. 使用 ASPE 加密
4. 计算密文 mAP
5. 验证 mAP 保持一致
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from core.aspe.dcmh_wrapper import ASPEForDCMH
from reference.DCMH.utils import calc_map_k


def generate_random_hash_codes(num_query, num_retrieval, bit_dim):
    """生成随机哈希码（模拟 DCMH 输出）"""
    qB = np.sign(np.random.randn(num_query, bit_dim)).astype(np.float64)
    rB = np.sign(np.random.randn(num_retrieval, bit_dim)).astype(np.float64)
    return qB, rB


def generate_labels(num_query, num_retrieval, num_labels, sparsity=0.3):
    """生成随机标签"""
    query_L = (np.random.rand(num_query, num_labels) > sparsity).astype(np.float64)
    retrieval_L = (np.random.rand(num_retrieval, num_labels) > sparsity).astype(np.float64)
    return query_L, retrieval_L


def main():
    print("=" * 60)
    print("DCMH + ASPE 集成演示")
    print("=" * 60)

    # 配置
    bit_dim = 64          # 哈希位数
    num_query = 100       # 查询数量
    num_retrieval = 1000  # 检索库大小
    num_labels = 20       # 标签数量
    seed = 42             # 随机种子

    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n[配置]")
    print(f"  哈希位数：{bit_dim}")
    print(f"  查询数量：{num_query}")
    print(f"  检索库大小：{num_retrieval}")
    print(f"  标签数量：{num_labels}")

    # 1. 生成哈希码（模拟 DCMH 输出）
    print(f"\n[步骤 1] 生成哈希码...")
    qB_img, rB_img = generate_random_hash_codes(num_query, num_retrieval, bit_dim)
    qB_txt, rB_txt = generate_random_hash_codes(num_query, num_retrieval, bit_dim)
    print(f"  图像查询哈希码：{qB_img.shape}")
    print(f"  图像检索哈希码：{rB_img.shape}")
    print(f"  文本查询哈希码：{qB_txt.shape}")
    print(f"  文本检索哈希码：{rB_txt.shape}")

    # 2. 生成标签
    print(f"\n[步骤 2] 生成标签...")
    query_L, retrieval_L = generate_labels(num_query, num_retrieval, num_labels)
    print(f"  查询标签：{query_L.shape}")
    print(f"  检索库标签：{retrieval_L.shape}")

    # 3. 计算原始 mAP（明文）
    print(f"\n[步骤 3] 计算原始 mAP（明文）...")

    # 图像→文本检索
    mapi2t_original = calc_map_k(
        torch.from_numpy(qB_img),
        torch.from_numpy(rB_txt),
        torch.from_numpy(query_L),
        torch.from_numpy(retrieval_L)
    )

    # 文本→图像检索
    mapt2i_original = calc_map_k(
        torch.from_numpy(qB_txt),
        torch.from_numpy(rB_img),
        torch.from_numpy(query_L),
        torch.from_numpy(retrieval_L)
    )

    print(f"  原始 mAP(图像→文本): {mapi2t_original:.6f}")
    print(f"  原始 mAP(文本→图像): {mapt2i_original:.6f}")

    # 4. ASPE 加密
    print(f"\n[步骤 4] ASPE 加密...")
    aspe = ASPEForDCMH(bit_dim=bit_dim, seed=seed)

    print("  加密检索库...")
    encrypted_rB_img = aspe.GenEnc(rB_img)
    encrypted_rB_txt = aspe.GenEnc(rB_txt)

    print("  生成查询陷阱门...")
    encrypted_qB_img = aspe.GenTrap(qB_img)
    encrypted_qB_txt = aspe.GenTrap(qB_txt)

    print(f"  加密检索库形状：{encrypted_rB_img.shape} (原始：{rB_img.shape})")
    print(f"  加密陷阱门形状：{encrypted_qB_img.shape} (原始：{qB_img.shape})")

    # 5. 计算密文 mAP
    print(f"\n[步骤 5] 计算密文 mAP...")

    mapi2t_aspe = aspe.calc_ciphertext_map(
        encrypted_qB_img, encrypted_rB_txt, query_L, retrieval_L
    )

    mapt2i_aspe = aspe.calc_ciphertext_map(
        encrypted_qB_txt, encrypted_rB_img, query_L, retrieval_L
    )

    print(f"  密文 mAP(图像→文本): {mapi2t_aspe:.6f}")
    print(f"  密文 mAP(文本→图像): {mapt2i_aspe:.6f}")

    # 6. 验证
    print(f"\n[步骤 6] 验证 mAP 一致性...")

    diff_i2t = abs(mapi2t_original - mapi2t_aspe)
    diff_t2i = abs(mapt2i_original - mapt2i_aspe)

    print(f"  mAP(图像→文本) 差异：{diff_i2t:.10f}")
    print(f"  mAP(文本→图像) 差异：{diff_t2i:.10f}")

    # 判断是否通过验证
    threshold = 1e-6
    i2t_pass = diff_i2t < threshold
    t2i_pass = diff_t2i < threshold

    print(f"\n[验证结果]")
    if i2t_pass and t2i_pass:
        print("  ✓ 通过！ASPE 加密前后 mAP 完全一致")
        print(f"    (差异 < {threshold})")
    else:
        print("  ✗ 失败！mAP 差异超过阈值")
        if not i2t_pass:
            print(f"    图像→文本差异：{diff_i2t:.10f}")
        if not t2i_pass:
            print(f"    文本→图像差异：{diff_t2i:.10f}")

    # 7. 不同哈希位数的比较
    print(f"\n[扩展测试] 不同哈希位数...")
    bit_configs = [16, 32, 64, 128]

    print(f"  {'位数':<6} {'原始 mAP(i→t)':<15} {'密文 mAP(i→t)':<15} {'差异':<12}")
    print(f"  {'-'*48}")

    for bit in bit_configs:
        # 生成新的哈希码
        qB_test, rB_test = generate_random_hash_codes(50, 500, bit)
        qL_test, rL_test = generate_labels(50, 500, 10)

        # 原始 mAP
        map_orig = calc_map_k(
            torch.from_numpy(qB_test),
            torch.from_numpy(rB_test),
            torch.from_numpy(qL_test),
            torch.from_numpy(rL_test)
        )

        # ASPE mAP
        aspe_test = ASPEForDCMH(bit_dim=bit, seed=seed)
        enc_rB = aspe_test.GenEnc(rB_test)
        enc_qB = aspe_test.GenTrap(qB_test)
        map_aspe = aspe_test.calc_ciphertext_map(enc_qB, enc_rB, qL_test, rL_test)

        diff = abs(map_orig - map_aspe)
        status = "✓" if diff < threshold else "✗"

        print(f"  {bit:<6} {map_orig:.6f}        {map_aspe:.6f}        {diff:.2e}     {status}")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)

    return i2t_pass and t2i_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
