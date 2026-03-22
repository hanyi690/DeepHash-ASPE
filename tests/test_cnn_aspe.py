"""
CNN+ASPE 加密检索功能验证脚本

验证 SkNN 加密方案的正确性：
1. 密钥生成
2. 内积保持性
3. 加密/解密流程
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from core.aspe.cnn_wrapper import ASPEForCNN


def test_inner_product_preservation():
    """测试 ASPE 内积保持性。"""
    print("=" * 60)
    print("测试 1: ASPE 内积保持性验证")
    print("=" * 60)

    # 初始化 ASPE
    aspe = ASPEForCNN(feature_dim=128, seed=42)
    aspe.generate_keys()

    # 生成随机特征向量
    p = torch.randn(128)
    q = torch.randn(128)

    # 明文内积
    plaintext_ip = torch.dot(p, q)
    print(f"\n明文内积：{plaintext_ip:.6f}")

    # 加密
    p_enc = aspe._encrypt_db_feature(p)
    q_enc = aspe._encrypt_query_feature(q)
    print(f"加密后维度：{p_enc.shape} (应该是 [256])")

    # 密文内积
    ciphertext_ip = torch.dot(p_enc, q_enc)
    print(f"密文内积：{ciphertext_ip:.6f}")

    # 计算差异
    diff = abs(plaintext_ip - ciphertext_ip)
    relative_diff = diff / (abs(plaintext_ip) + 1e-10)

    print(f"\n绝对差异：{diff:.2e}")
    print(f"相对差异：{relative_diff:.2e}")

    # 验证
    passed = relative_diff < 1e-3
    print(f"\n验证结果：{'通过' if passed else '失败'}")

    return passed


def test_encryption_workflow():
    """测试完整的加密检索流程。"""
    print("\n" + "=" * 60)
    print("测试 2: 完整加密检索流程")
    print("=" * 60)

    # 初始化
    aspe = ASPEForCNN(feature_dim=64, seed=42)
    aspe.generate_keys()

    # 模拟数据库特征
    db_size = 100
    db_features = torch.randn(db_size, 64)

    # 加密数据库
    print("\n加密数据库...")
    encrypted_db = aspe.encrypt_database(db_features)
    print(f"加密数据库维度：{encrypted_db.shape} (应该是 [100, 128])")

    # 模拟查询
    query = torch.randn(64)
    print("\n执行加密检索...")

    # 加密检索
    scores, indices = aspe.search(encrypted_db, query, top_k=5)

    print(f"Top-5 索引：{indices.tolist()}")
    print(f"Top-5 分数：{['{:.4f}'.format(s) for s in scores.tolist()]}")

    # 验证明文检索结果一致
    print("\n验证明文检索...")
    plaintext_scores = torch.matmul(db_features, query)
    _, plaintext_indices = torch.topk(plaintext_scores, k=5)

    print(f"明文 Top-5 索引：{plaintext_indices.tolist()}")

    # 比较
    same = torch.equal(indices, plaintext_indices)
    print(f"\n结果一致性：{'一致' if same else '不一致'}")

    return True


def test_key_save_load():
    """测试密钥保存和加载。"""
    print("\n" + "=" * 60)
    print("测试 3: 密钥保存/加载")
    print("=" * 60)

    import tempfile
    import os

    # 生成密钥
    aspe1 = ASPEForCNN(feature_dim=64, seed=None)
    aspe1.generate_keys()

    # 保存到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
        temp_path = f.name

    try:
        aspe1.save_keys(temp_path)
        print(f"密钥已保存：{temp_path}")

        # 加载到新实例
        aspe2 = ASPEForCNN(feature_dim=64)
        aspe2.load_keys(temp_path)

        # 验证密钥相同
        m1_same = torch.allclose(aspe1.M1, aspe2.M1)
        m2_same = torch.allclose(aspe1.M2, aspe2.M2)
        s_same = torch.allclose(aspe1.S, aspe2.S)

        print(f"M1 一致：{m1_same}")
        print(f"M2 一致：{m2_same}")
        print(f"S 一致：{s_same}")

        all_same = m1_same and m2_same and s_same
        print(f"\n密钥一致性：{'一致' if all_same else '不一致'}")

        return all_same

    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main():
    """运行所有测试。"""
    print("\n" + "=" * 60)
    print("CNN+ASPE 加密检索功能验证")
    print("=" * 60)

    results = []

    # 测试 1：内积保持性
    results.append(("内积保持性", test_inner_product_preservation()))

    # 测试 2：完整流程
    results.append(("完整检索流程", test_encryption_workflow()))

    # 测试 3：密钥保存/加载
    results.append(("密钥保存/加载", test_key_save_load()))

    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")

    print(f"\n总计：{passed}/{total} 通过")

    if passed == total:
        print("\n🎉 所有测试通过！CNN+ASPE 功能正常。")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查问题。")


if __name__ == "__main__":
    main()
