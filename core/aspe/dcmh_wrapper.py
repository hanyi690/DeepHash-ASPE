"""
DCMH 哈希码加密包装器

基于 SIGMOD'09 论文实现 ASPE 算法（SkNN 风格），密文内积 = 明文内积。
参考：https://blog.csdn.net/qq_36458536/article/details/128367624

核心原理：
- GenEnc（检索库加密）：S[i]=0 复制，S[i]=1 随机拆分
- GenTrap（查询加密）：S[i]=1 复制，S[i]=0 固定 r=0 拆分
- 密文内积 = v1·w1 + v2·w2 = v·w（明文内积）
"""

import numpy as np
import torch
from typing import Optional, Tuple


class ASPEForDCMH:
    """
    ASPE 加密与 DCMH 哈希码的接口类。

    将 DCMH 生成的 {-1, +1} 哈希码加密，密文内积 = 明文内积。

    核心洞察：
    1. 汉明距离与内积的线性关系：hamm(B1, B2) = 0.5 × (bit - B1·B2)
    2. ASPE 内积保持性：密文内积 = 明文内积
    3. 排序不变性：汉明距离排序 = 密文内积排序
    """

    def __init__(self, bit_dim: int, seed: int = 42):
        """
        初始化 ASPE for DCMH。

        参数：
            bit_dim: 哈希码位数（如 16, 32, 64）
            seed: 随机种子
        """
        self.bit = bit_dim
        self.seed = seed

        # 生成密钥 M1, M2, S
        np.random.seed(seed)
        d = bit_dim

        # 生成可逆矩阵 M1, M2
        self.M1 = np.random.randn(d, d)
        self.M2 = np.random.randn(d, d)

        # 确保矩阵可逆
        while np.abs(np.linalg.det(self.M1)) < 1e-10:
            self.M1 = np.random.randn(d, d)
        while np.abs(np.linalg.det(self.M2)) < 1e-10:
            self.M2 = np.random.randn(d, d)

        # 生成二元拆分向量 S
        self.S = np.random.randint(0, 2, d)

        # 预计算逆矩阵
        self.M1_inv = np.linalg.inv(self.M1)
        self.M2_inv = np.linalg.inv(self.M2)

    def GenEnc(self, codes: np.ndarray) -> np.ndarray:
        """
        加密检索库哈希码。

        规则：
        - S[i]=0: 复制 v1[i] = v2[i] = v[i]
        - S[i]=1: 随机拆分 v1[i] + v2[i] = v[i]

        参数：
            codes: {-1, +1}^{N×bit} 检索库哈希码

        返回：
            {N×(2×bit)} 加密检索库（两个份额水平拼接）
        """
        if isinstance(codes, torch.Tensor):
            codes = codes.cpu().numpy()

        codes = codes.astype(np.float64)
        N, d = codes.shape

        # 随机拆分值
        r = np.random.randn(N, d)

        # S=0 复制，S=1 随机拆分
        # v1 = where(S=1, r, v)
        # v2 = where(S=1, v-r, v)
        v1 = np.where(self.S == 1, r, codes)
        v2 = np.where(self.S == 1, codes - r, codes)

        # 矩阵变换：M^T @ v
        enc1 = v1 @ self.M1.T
        enc2 = v2 @ self.M2.T

        return np.hstack([enc1, enc2])

    def GenTrap(self, codes: np.ndarray) -> np.ndarray:
        """
        加密查询哈希码（生成陷阱门）。

        规则：
        - S[i]=1: 复制 w1[i] = w2[i] = w[i]
        - S[i]=0: 固定 r=0 拆分 w1[i] = 0, w2[i] = w[i]（数值稳定）

        参数：
            codes: {-1, +1}^{M×bit} 查询哈希码

        返回：
            {M×(2×bit)} 加密查询（陷阱门，两个份额水平拼接）
        """
        if isinstance(codes, torch.Tensor):
            codes = codes.cpu().numpy()

        codes = codes.astype(np.float64)
        N, d = codes.shape

        # S=1 复制，S=0 固定 r=0 拆分
        # w1 = where(S=0, 0, w)
        # w2 = where(S=0, w, w) = w
        w1 = np.where(self.S == 0, 0.0, codes)
        w2 = codes.copy()

        # 逆矩阵变换：w @ M^(-1)（不是 M^(-1).T）
        trap1 = w1 @ self.M1_inv
        trap2 = w2 @ self.M2_inv

        return np.hstack([trap1, trap2])

    def ciphertext_hamming_distance(self,
                                    encrypted_q: np.ndarray,
                                    encrypted_r: np.ndarray) -> np.ndarray:
        """
        计算密文汉明距离。

        密文内积 = 明文内积，汉明距离 = 0.5 × (bit - 内积)

        参数：
            encrypted_q: 加密查询 [M, 2×bit] 或 [2×bit]
            encrypted_r: 加密检索库 [N, 2×bit]

        返回：
            汉明距离数组 [M, N]
        """
        if isinstance(encrypted_q, torch.Tensor):
            encrypted_q = encrypted_q.cpu().numpy()
        if isinstance(encrypted_r, torch.Tensor):
            encrypted_r = encrypted_r.cpu().numpy()

        if encrypted_q.ndim == 1:
            encrypted_q = encrypted_q.reshape(1, -1)

        d = self.bit
        q1, q2 = encrypted_q[:, :d], encrypted_q[:, d:]
        r1, r2 = encrypted_r[:, :d], encrypted_r[:, d:]

        # 密文内积 = 明文内积
        inner_products = q1 @ r1.T + q2 @ r2.T

        # 汉明距离 = 0.5 × (bit - 内积)
        hamm = 0.5 * (self.bit - inner_products)

        return np.clip(hamm, 0, self.bit)

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
            encrypted_qB: 加密查询哈希码 [num_query, 2×bit]
            encrypted_rB: 加密检索库哈希码 [num_retrieval, 2×bit]
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
            q_L = query_L[iter_idx:iter_idx+1]

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
            # 四舍五入消除浮点误差，确保相同距离的值排序一致
            hamm_rounded = np.round(hamm, decimals=10)
            # 使用 lexsort 保证确定性：先按距离，再按索引
            ind = np.lexsort((np.arange(len(hamm_rounded)), hamm_rounded))

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

    def get_key_info(self) -> dict:
        """
        获取密钥信息（用于调试和验证）。

        返回：
            包含密钥信息的字典
        """
        return {
            'bit': self.bit,
            'seed': self.seed,
            'M1_shape': self.M1.shape,
            'M2_shape': self.M2.shape,
            'S_shape': self.S.shape
        }

    def verify_inner_product_preservation(self,
                                          codes: np.ndarray,
                                          num_samples: int = 5) -> dict:
        """
        验证密文内积 = 明文内积。

        参数：
            codes: 哈希码数组
            num_samples: 采样数量

        返回：
            包含验证结果的字典
        """
        if isinstance(codes, torch.Tensor):
            codes = codes.cpu().numpy()

        codes = codes.astype(np.float64)
        n = min(num_samples, len(codes))

        encrypted = self.GenEnc(codes[:n])

        results = {
            'preserved': True,
            'max_error': 0.0,
            'details': []
        }

        d = self.bit
        for i in range(n):
            for j in range(n):
                # 明文内积
                plain_ip = np.dot(codes[i], codes[j])

                # 密文内积
                e1, e2 = encrypted[i, :d], encrypted[i, d:]
                f1, f2 = encrypted[j, :d], encrypted[j, d:]
                cipher_ip = np.dot(e1, f1) + np.dot(e2, f2)

                error = abs(plain_ip - cipher_ip)
                results['max_error'] = max(results['max_error'], error)

                if error > 1e-6:
                    results['preserved'] = False

                results['details'].append({
                    'i': i, 'j': j,
                    'plain_ip': plain_ip,
                    'cipher_ip': cipher_ip,
                    'error': error
                })

        return results

    def verify_sorting_consistency(self,
                                   query_codes: np.ndarray,
                                   retrieval_codes: np.ndarray,
                                   top_k: list = [10, 50, 100],
                                   thresholds: list = [0.90, 0.85, 0.80]) -> dict:
        """
        验证明文和密文排序一致性。

        参数：
            query_codes: 查询哈希码 {-1, +1}^{M×bit}
            retrieval_codes: 检索库哈希码 {-1, +1}^{N×bit}
            top_k: 要验证的前K个结果列表
            thresholds: 每个K值对应的阈值（交集比例）

        返回：
            包含验证结果的字典：
            - passed: 是否通过所有验证
            - details: 每个查询的详细结果
            - overlap_ratios: 各K值的平均交集比例
        """
        if isinstance(query_codes, torch.Tensor):
            query_codes = query_codes.cpu().numpy()
        if isinstance(retrieval_codes, torch.Tensor):
            retrieval_codes = retrieval_codes.cpu().numpy()

        query_codes = query_codes.astype(np.float64)
        retrieval_codes = retrieval_codes.astype(np.float64)

        # 加密
        encrypted_q = self.GenTrap(query_codes)
        encrypted_r = self.GenEnc(retrieval_codes)

        # 计算明文汉明距离
        plain_inner = query_codes @ retrieval_codes.T
        plain_hamm = 0.5 * (self.bit - plain_inner)

        # 计算密文汉明距离
        cipher_hamm = self.ciphertext_hamming_distance(encrypted_q, encrypted_r)

        # 初始化结果
        results = {
            'passed': True,
            'overlap_ratios': {k: [] for k in top_k},
            'distance_errors': [],
            'details': []
        }

        num_queries = len(query_codes)

        for i in range(num_queries):
            # 明文排序（使用与密文一致的排序逻辑）
            plain_h = plain_hamm[i]
            plain_rounded = np.round(plain_h, decimals=10)
            plain_ind = np.lexsort((np.arange(len(plain_rounded)), plain_rounded))

            # 密文排序
            cipher_h = cipher_hamm[i]
            cipher_rounded = np.round(cipher_h, decimals=10)
            cipher_ind = np.lexsort((np.arange(len(cipher_rounded)), cipher_rounded))

            # 计算距离误差
            dist_error = np.abs(plain_h - cipher_h).max()
            results['distance_errors'].append(dist_error)

            # 计算各K值的交集比例
            detail = {'query_idx': i, 'overlaps': {}}
            for k, threshold in zip(top_k, thresholds):
                plain_top_k = set(plain_ind[:k])
                cipher_top_k = set(cipher_ind[:k])
                overlap = len(plain_top_k & cipher_top_k) / k
                results['overlap_ratios'][k].append(overlap)
                detail['overlaps'][k] = overlap

                if overlap < threshold:
                    results['passed'] = False

            results['details'].append(detail)

        # 计算平均交集比例
        for k in top_k:
            results['overlap_ratios'][f'{k}_mean'] = np.mean(results['overlap_ratios'][k])

        return results


if __name__ == "__main__":
    # 测试 ASPE for DCMH
    print("测试 ASPE for DCMH (SkNN 风格)")
    print("=" * 60)

    # 初始化
    bit = 64
    aspe = ASPEForDCMH(bit_dim=bit)

    print(f"哈希码位数: {bit}")
    print(f"密文维度: {2 * bit}")
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

    # 验证内积保持性
    print("\n验证内积保持性...")
    verification = aspe.verify_inner_product_preservation(retrieval_codes)
    print(f"内积保持: {'是' if verification['preserved'] else '否'}")
    print(f"最大误差: {verification['max_error']:.2e}")

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

        # 计算距离误差
        error = np.abs(plain_hamm - cipher_hamm).max()

        # 计算排序相关性
        rank_plain = np.argsort(plain_hamm)
        rank_cipher = np.argsort(cipher_hamm)

        # 计算前 10 的交集
        overlap = len(set(rank_plain[:10]) & set(rank_cipher[:10])) / 10

        print(f"  查询 {i+1}: 最大距离误差 = {error:.2e}, 前 10 交集比例 = {overlap:.2f}")

    print("\n测试完成!")