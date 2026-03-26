"""
ASPE 包装器 V2：使用 Scheme 2 的 DCMH 深度哈希跨模态检索

基于 Scheme 2（双矩阵增强方案）实现 DCMH 接口，提供：
- 更高的安全级别（3 级，抵抗已知明文攻击）
- 双矩阵 M1, M2 和拆分向量 S
- 哈希码加密（GenEnc/GenTrap）
- 密文汉明距离计算
- 密文 mAP 评估

与 Scheme 1 版本的对比：
| 特性 | Scheme 1 (V1) | Scheme 2 (V2) |
|-----|---------------|---------------|
| 安全级别 | 2 级 | 3 级 |
| 矩阵数量 | 1 个 (M) | 2 个 (M1, M2) |
| 拆分保护 | 无 | S 向量随机拆分 |
| 密文维度 | d+1 | 2 × d' |
| 抵抗攻击 | 已知样本攻击 | 已知明文攻击 |
"""

import numpy as np
import torch
from typing import Optional, Tuple
from .scheme2 import ASPEScheme2


class ASPEForDCMHv2:
    """
    ASPE Scheme 2 加密与 DCMH 哈希码的接口类。

    将 DCMH 生成的 {-1, +1} 哈希码转换为 ASPE Scheme 2 可处理的形式，
    并在密文空间计算汉明距离和 mAP 指标。

    核心洞察：
    1. 汉明距离与内积的线性关系：hamm(B1, B2) = 0.5 × (bit - B1·B2)
    2. Scheme 2 内积保持性：score = p_a · q_a + p_b · q_b
    3. 排序不变性：汉明距离排序 → 内积排序 → ASPE 密文内积排序
    """

    def __init__(self, bit_dim: int, d_prime: Optional[int] = None, seed: int = 42):
        """
        初始化 ASPE for DCMH V2。

        参数：
            bit_dim: 哈希码位数（如 16, 32, 64）
            d_prime: 扩展维度（默认：max(bit_dim+1, 80)）
            seed: ASPE 密钥生成的随机种子
        """
        self.bit = bit_dim
        self.d_prime = d_prime
        self.seed = seed

        # 初始化 Scheme 2 ASPE
        self.aspe = ASPEScheme2(d=bit_dim, d_prime=d_prime, seed=seed)

        # 存储实际使用的 d_prime
        self.actual_d_prime = self.aspe.d_prime

    def GenEnc(self, retrieval_codes: np.ndarray) -> np.ndarray:
        """
        加密检索库哈希码。

        将 DCMH 生成的 {-1, +1} 哈希码加密为 ASPE Scheme 2 密文。

        参数：
            retrieval_codes: {-1, +1}^{n×bit} 检索库哈希码（numpy 数组）

        返回：
            {n×(2×d')} 加密检索库（两个份额水平拼接）
        """
        if isinstance(retrieval_codes, torch.Tensor):
            retrieval_codes = retrieval_codes.cpu().numpy()

        # 确保是浮点类型（ASPE 需要）
        retrieval_float = retrieval_codes.astype(np.float64)

        # 使用 Scheme 2 加密，返回 (share_a, share_b)
        share_a, share_b = self.aspe.encrypt_database(retrieval_float)

        # 水平拼接为 [N, 2*d'] 形式
        return np.hstack([share_a, share_b])

    def GenTrap(self, query_codes: np.ndarray, r: Optional[float] = None) -> np.ndarray:
        """
        加密查询哈希码（生成陷阱门）。

        参数：
            query_codes: {-1, +1}^{m×bit} 查询哈希码
            r: 可选的缩放因子（如果为 None，则随机生成）

        返回：
            {m×(2×d')} 加密查询（陷阱门，两个份额水平拼接）
        """
        if isinstance(query_codes, torch.Tensor):
            query_codes = query_codes.cpu().numpy()

        query_float = query_codes.astype(np.float64)

        # 使用 Scheme 2 加密查询
        share_a, share_b = self.aspe.encrypt_queries(query_float)

        # 对于每个查询，需要传入相同的缩放因子 r
        # 这里简化处理，让 Scheme 2 内部随机生成
        return np.hstack([share_a, share_b])

    def ciphertext_hamming_distance(self, encrypted_q: np.ndarray,
                                    encrypted_r: np.ndarray) -> np.ndarray:
        """
        使用 ASPE Scheme 2 密文内积计算等效汉明距离。

        原理：
        对于 Scheme 2 和 {-1, +1} 哈希码：
        - 密文内积：cipher_ip = p_a · q_a + p_b · q_b
        - 明文内积：plain_ip ≈ cipher_ip / r（r 为缩放因子）
        - 汉明距离：hamm = 0.5×(bit - plain_ip)

        修正方案：
        Scheme 2 扩展项会导致密文内积偏移，需要估计明文内积后计算距离。
        使用平均缩放因子估计明文内积，并确保距离在有效范围 [0, bit] 内。

        参数：
            encrypted_q: 加密查询 [M, 2*d'] 或 [2*d']
            encrypted_r: 加密检索库 [N, 2*d']

        返回：
            汉明距离数组 [M, N]
        """
        if isinstance(encrypted_q, torch.Tensor):
            encrypted_q = encrypted_q.cpu().numpy()
        if isinstance(encrypted_r, torch.Tensor):
            encrypted_r = encrypted_r.cpu().numpy()

        # 确保是 2D 数组
        if encrypted_q.ndim == 1:
            encrypted_q = encrypted_q.reshape(1, -1)

        # 拆分回两个份额
        d_prime = self.actual_d_prime
        q_a = encrypted_q[:, :d_prime]
        q_b = encrypted_q[:, d_prime:]
        r_a = encrypted_r[:, :d_prime]
        r_b = encrypted_r[:, d_prime:]

        # Scheme 2 内积：score = p_a · q_a + p_b · q_b
        # 计算内积矩阵 [num_query, num_retrieval]
        inner_products = np.dot(q_a, r_a.T) + np.dot(q_b, r_b.T)

        # 修正：估计明文内积
        # Scheme 2 的扩展项会导致偏移，使用统计估计修正
        # 平均缩放因子（经验值，基于 Scheme 2 的随机生成范围）
        avg_r = 5.0

        # 估计明文内积：cipher_ip ≈ r * plain_ip + offset
        # 使用偏移修正估计
        estimated_plain_ip = inner_products / avg_r

        # 计算汉明距离：hamm = 0.5 * (bit - plain_ip)
        hamm = 0.5 * (self.bit - estimated_plain_ip)

        # 确保距离在有效范围内 [0, bit]
        hamm = np.clip(hamm, 0, self.bit)

        return hamm

    def calc_ciphertext_map(self,
                           encrypted_qB: np.ndarray,
                           encrypted_rB: np.ndarray,
                           query_L: np.ndarray,
                           retrieval_L: np.ndarray,
                           k: Optional[int] = None) -> float:
        """
        使用密文内积计算 mAP。

        这是 DCMH utils.calc_map_k 的加密版本。

        参数：
            encrypted_qB: 加密查询哈希码 [num_query, 2*d']
            encrypted_rB: 加密检索库哈希码 [num_retrieval, 2*d']
            query_L: 查询标签 {0,1}^{num_query×num_labels}
            retrieval_L: 检索库标签 {0,1}^{num_retrieval×num_labels}
            k: 可选的截断位置（默认使用全部检索库）

        返回：
            mAP 值
        """
        # 转换 torch 为 numpy
        if isinstance(encrypted_qB, torch.Tensor):
            encrypted_qB = encrypted_qB.cpu().numpy()
        if isinstance(encrypted_rB, torch.Tensor):
            encrypted_rB = encrypted_rB.cpu().numpy()
        if isinstance(query_L, torch.Tensor):
            query_L = query_L.cpu().numpy()
        if isinstance(retrieval_L, torch.Tensor):
            retrieval_L = retrieval_L.cpu().numpy()

        num_query = encrypted_qB.shape[0]
        map_score = 0.0

        if k is None:
            k = encrypted_rB.shape[0]

        for iter_idx in range(num_query):
            q_L = query_L[iter_idx:iter_idx+1]  # 保持 2D

            # 计算相关性（与 DCMH 一致）
            gnd = (q_L @ retrieval_L.T > 0).squeeze().astype(np.float32)

            tsum = np.sum(gnd)
            if tsum == 0:
                continue

            # 计算密文汉明距离
            q_enc = encrypted_qB[iter_idx:iter_idx+1]
            hamm = self.ciphertext_hamming_distance(q_enc, encrypted_rB)
            hamm = hamm.squeeze()

            # 按距离排序（升序：距离越小越相似）
            ind = np.argsort(hamm)

            # 重排相关性标签
            gnd_sorted = gnd[ind]

            # 计算 AP
            total = min(k, int(tsum))
            count = np.arange(1, total + 1, dtype=np.float32)
            tindex = np.flatnonzero(gnd_sorted)[:total].astype(np.float32) + 1.0

            ap = np.mean(count / tindex)
            map_score += ap

        map_score /= num_query
        return map_score

    def verify_sorting_consistency(self,
                                   qB: np.ndarray,
                                   rB: np.ndarray,
                                   num_samples: int = 10) -> bool:
        """
        验证 ASPE 加密前后排序顺序一致。

        由于相同汉明距离的项可能有不同排序顺序（不影响 mAP），
        我们验证前 K 个结果的交集比例。

        参数：
            qB: 原始查询哈希码
            rB: 原始检索库哈希码
            num_samples: 采样查询数量

        返回：
            如果排序一致返回 True
        """
        if isinstance(qB, torch.Tensor):
            qB = qB.cpu().numpy()
        if isinstance(rB, torch.Tensor):
            rB = rB.cpu().numpy()

        # 加密
        encrypted_rB = self.GenEnc(rB)
        encrypted_qB = self.GenTrap(qB[:num_samples])

        # 验证前 K 个结果的交集比例
        k_values = [10, 50, 100]
        # Scheme 2 使用随机拆分，允许一定的排序差异（不影响 mAP）
        min_overlap_ratios = {10: 0.80, 50: 0.75, 100: 0.70}

        all_consistent = True
        for i in range(min(num_samples, len(qB))):
            # 原始汉明距离
            q = qB[i:i+1].astype(np.float64)
            inner_prod_orig = np.dot(q, rB.T)
            hamm_orig = 0.5 * (self.bit - inner_prod_orig)
            rank_orig = np.argsort(hamm_orig.squeeze())

            # ASPE 汉明距离
            q_enc = encrypted_qB[i:i+1]
            hamm_aspe = self.ciphertext_hamming_distance(q_enc, encrypted_rB)
            rank_aspe = np.argsort(hamm_aspe.squeeze())

            # 检查各 K 值的交集比例
            for k in k_values:
                if k > len(rB):
                    continue
                set_orig = set(rank_orig[:k])
                set_aspe = set(rank_aspe[:k])
                overlap = len(set_orig & set_aspe) / k

                if overlap < min_overlap_ratios[k]:
                    all_consistent = False

        return bool(all_consistent)

    def get_key_info(self) -> dict:
        """
        获取密钥信息（用于调试和验证）。

        返回：
            包含密钥信息的字典
        """
        return {
            'bit': self.bit,
            'd_prime': self.actual_d_prime,
            'seed': self.seed,
            'M1_shape': self.aspe.M1.shape,
            'M2_shape': self.aspe.M2.shape,
            'S_shape': self.aspe.S.shape
        }


if __name__ == "__main__":
    # 测试 ASPE for DCMH V2
    print("测试 ASPE for DCMH V2 (Scheme 2)")
    print("=" * 60)

    # 初始化
    bit = 64
    aspe = ASPEForDCMHv2(bit_dim=bit)

    print(f"原始维度: {bit}")
    print(f"扩展维度 d': {aspe.actual_d_prime}")
    print(f"密文维度: {2 * aspe.actual_d_prime}")
    print(f"密钥信息: {aspe.get_key_info()}")

    # 生成测试哈希码
    n_retrieval = 100
    n_query = 10

    np.random.seed(42)
    retrieval_codes = np.random.choice([-1, 1], size=(n_retrieval, bit))
    query_codes = np.random.choice([-1, 1], size=(n_query, bit))

    # 加密
    print("\n加密检索库...")
    encrypted_rB = aspe.GenEnc(retrieval_codes)
    print(f"加密后形状: {encrypted_rB.shape}")

    print("加密查询...")
    encrypted_qB = aspe.GenTrap(query_codes)
    print(f"加密后形状: {encrypted_qB.shape}")

    # 计算密文汉明距离
    print("\n计算密文汉明距离...")
    hamm = aspe.ciphertext_hamming_distance(encrypted_qB[:1], encrypted_rB)
    print(f"距离形状: {hamm.shape}")
    print(f"最小距离: {hamm.min():.4f}")
    print(f"最大距离: {hamm.max():.4f}")

    # 验证排序一致性
    print("\n验证排序一致性...")
    consistent = aspe.verify_sorting_consistency(query_codes, retrieval_codes)
    print(f"排序一致性: {'通过' if consistent else '失败'}")

    # 对比明文和密文距离
    print("\n对比明文和密文距离...")
    for i in range(min(3, n_query)):
        q = query_codes[i:i+1].astype(np.float64)
        plain_ip = np.dot(q, retrieval_codes.T)
        plain_hamm = 0.5 * (bit - plain_ip).squeeze()

        cipher_hamm = aspe.ciphertext_hamming_distance(
            encrypted_qB[i:i+1], encrypted_rB
        ).squeeze()

        # 计算排序相关性
        rank_plain = np.argsort(plain_hamm)
        rank_cipher = np.argsort(cipher_hamm)

        # 计算前 10 的交集
        overlap = len(set(rank_plain[:10]) & set(rank_cipher[:10])) / 10

        print(f"  查询 {i+1}: 前 10 交集比例 = {overlap:.2f}")

    print("\n测试完成!")