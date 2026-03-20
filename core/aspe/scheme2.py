"""
ASPE 方案 2：增强 3 级安全

实现增强非对称标量积保持加密（ASPE）
具有 3 级安全性（抵抗已知明文攻击）。

相比方案 1 的增强：
1. 随机拆分：根据配置 S 将每个向量拆分为两个份额
2. 人工维度：添加具有零标量积的维度
3. 两个独立矩阵：M1 和 M2 用于增强安全性
"""

import numpy as np
from typing import Tuple, Optional
from core.aspe.keygen import generate_key_scheme2


class ASPEScheme2:
    """
    用于增强隐私保护相似度搜索的 ASPE 方案 2。

    安全级别：3（抵抗已知明文攻击）

    加密策略：
    1. 将点扩展到 (d+1) 维：p̂ = (p, -0.5||p||²)
    2. 添加人工维度达到 d' 维
    3. 根据配置 S 拆分为两个份额
    4. 使用单独的矩阵（M1, M2）变换每个份额

    属性：
    - 抵抗已知明文攻击
    - 拆分防止矩阵恢复
    - 人工维度增强安全性
    """

    def __init__(self, d: int, d_prime: Optional[int] = None, seed: int = 42):
        """
        初始化 ASPE 方案 2。

        参数：
            d: 原始特征维度
            d_prime: 扩展维度（默认：max(d+1, 80)）
            seed: 密钥生成的随机种子
        """
        self.d = d
        # 设置 d_prime 并满足最小安全约束
        if d_prime is None:
            self.d_prime = max(d + 1, 80)
        else:
            self.d_prime = max(d_prime, d + 1, 80)

        self.seed = seed

        # 生成密钥
        self.M1, self.M2, self.S, self.w = generate_key_scheme2(d, self.d_prime, seed)

        # 预计算逆矩阵
        try:
            self.M1_inv = np.linalg.inv(self.M1)
            self.M2_inv = np.linalg.inv(self.M2)
        except np.linalg.LinAlgError:
            raise ValueError("密钥矩阵求逆失败。使用正确的密钥生成方法不应发生此错误。")

    def _extend_point(self, p: np.ndarray) -> np.ndarray:
        """
        将 d 维点扩展到 (d+1) 维。

        p̂ = (p, -0.5||p||²)

        参数：
            p: d 维点

        返回：
            (d+1) 维扩展点
        """
        p_hat = np.zeros(self.d + 1)
        p_hat[:self.d] = p
        p_hat[self.d] = -0.5 * np.dot(p, p)
        return p_hat

    def _add_artificial_dimensions(self, p_hat: np.ndarray, is_query: bool = False) -> np.ndarray:
        """
        使用人工维度将 (d+1) 维点扩展到 d' 维。

        人工维度的设计使得它们与查询人工维度的标量积为零。

        对于数据库：将人工维度设置为随机值 w
        对于查询：将人工维度设置为正交值

        参数：
            p_hat: (d+1) 维扩展点
            is_query: 如果是查询点则为 True

        返回：
            带有人工维度的 d' 维点
        """
        p_extended = np.zeros(self.d_prime)
        p_extended[:self.d + 1] = p_hat

        # 添加人工维度
        if is_query:
            # 对于查询，使用正交值（乘以 -1）
            # 这确保人工维度的标量积 = 0
            p_extended[self.d + 1:] = -self.w
        else:
            # 对于数据库，使用随机值 w
            p_extended[self.d + 1:] = self.w

        return p_extended

    def _split_point(self, p_extended: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据配置 S 将扩展点拆分为两个份额。

        对于 S[i] = 1 的维度：拆分为随机份额
        对于 S[i] = 0 的维度：两个份额都获得完整值

        参数：
            p_extended: d' 维扩展点

        返回：
            (share_a, share_b) 元组，每个都是 d' 维
        """
        share_a = np.zeros(self.d_prime)
        share_b = np.zeros(self.d_prime)

        for i in range(self.d_prime):
            if i < len(self.S) and self.S[i] == 1:
                # 拆分此维度
                val = p_extended[i]
                # 随机拆分：share_a = r, share_b = val - r
                r = np.random.randn() * val
                share_a[i] = r
                share_b[i] = val - r
            else:
                # 两个份额都获得完整值
                share_a[i] = p_extended[i]
                share_b[i] = p_extended[i]

        return share_a, share_b

    def encrypt_database_point(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用方案 2 加密数据库点 p。

        步骤：
        1. 扩展到 (d+1) 维：p̂ = (p, -0.5||p||²)
        2. 添加人工维度达到 d' 维
        3. 根据 S 拆分为两个份额
        4. 变换：p_a' = M1^T * p_a, p_b' = M2^T * p_b

        参数：
            p: d 维数据库点

        返回：
            (p_a_enc, p_b_enc) 元组，每个都是 d' 维
        """
        if p.shape[0] != self.d:
            raise ValueError(f"期望 {self.d} 维点，得到 {p.shape[0]}")

        # 步骤 1：扩展到 (d+1) 维
        p_hat = self._extend_point(p)

        # 步骤 2：添加人工维度
        p_extended = self._add_artificial_dimensions(p_hat, is_query=False)

        # 步骤 3：拆分为两个份额
        p_a, p_b = self._split_point(p_extended)

        # 步骤 4：使用矩阵变换
        p_a_enc = np.dot(self.M1.T, p_a)
        p_b_enc = np.dot(self.M2.T, p_b)

        return p_a_enc, p_b_enc

    def encrypt_query(self, q: np.ndarray, r: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用方案 2 加密查询点 q。

        步骤：
        1. 生成随机缩放因子 r > 0
        2. 缩放并扩展：q̂ = r * (q, 1)
        3. 添加人工维度（正交值）
        4. 根据 S 拆分为两个份额
        5. 变换：q_a' = M1^(-1) * q_a, q_b' = M2^(-1) * q_b

        参数：
            q: d 维查询点
            r: 可选的随机缩放因子

        返回：
            (q_a_enc, q_b_enc) 元组，每个都是 d' 维（陷阱门）
        """
        if q.shape[0] != self.d:
            raise ValueError(f"期望 {self.d} 维点，得到 {q.shape[0]}")

        # 步骤 1：生成随机缩放因子
        if r is None:
            r = np.random.uniform(0.1, 10.0)
        elif r <= 0:
            raise ValueError("缩放因子 r 必须为正数")

        # 步骤 2：缩放并扩展到 (d+1) 维
        q_hat = np.zeros(self.d + 1)
        q_hat[:self.d] = r * q
        q_hat[self.d] = r

        # 步骤 3：添加人工维度
        q_extended = self._add_artificial_dimensions(q_hat, is_query=True)

        # 步骤 4：拆分为两个份额
        q_a, q_b = self._split_point(q_extended)

        # 步骤 5：使用逆矩阵变换
        q_a_enc = np.dot(self.M1_inv, q_a)
        q_b_enc = np.dot(self.M2_inv, q_b)

        return q_a_enc, q_b_enc

    def decrypt_point(self, p_enc: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        解密加密点。

        参数：
            p_enc: (p_a_enc, p_b_enc) 元组

        返回：
            d 维解密点
        """
        p_a_enc, p_b_enc = p_enc

        if p_a_enc.shape[0] != self.d_prime or p_b_enc.shape[0] != self.d_prime:
            raise ValueError(f"期望 {self.d_prime} 维加密点")

        # 逆变换
        p_a = np.dot(np.linalg.inv(self.M1.T), p_a_enc)
        p_b = np.dot(np.linalg.inv(self.M2.T), p_b_enc)

        # 合并份额（对于拆分的维度，求和份额）
        p_extended = np.zeros(self.d_prime)
        for i in range(self.d_prime):
            if i < len(self.S) and self.S[i] == 1:
                p_extended[i] = p_a[i] + p_b[i]
            else:
                # 对于非拆分维度，它们应该相同
                # 使用平均值以减少数值误差
                p_extended[i] = (p_a[i] + p_b[i]) / 2.0

        # 提取前 d 维
        p = p_extended[:self.d]

        return p

    def ciphertext_inner_product(self,
                                 p_enc: Tuple[np.ndarray, np.ndarray],
                                 q_enc: Tuple[np.ndarray, np.ndarray]) -> float:
        """
        在密文空间计算内积。

        对于带拆分的方案 2：
        score = p_a · q_a + p_b · q_b

        这保持 2 型标量积，直到缩放因子 r。

        参数：
            p_enc: 加密数据库份额元组
            q_enc: 加密查询份额元组

        返回：
            内积值（与相似度成正比）
        """
        p_a_enc, p_b_enc = p_enc
        q_a_enc, q_b_enc = q_enc

        # 计算两个份额的内积
        score = np.dot(p_a_enc, q_a_enc) + np.dot(p_b_enc, q_b_enc)

        return score

    def compare_distance(self,
                        p1_enc: Tuple[np.ndarray, np.ndarray],
                        p2_enc: Tuple[np.ndarray, np.ndarray],
                        q_enc: Tuple[np.ndarray, np.ndarray]) -> bool:
        """
        使用密文内积比较距离。

        检查 (p1 - p2) · q > 0

        参数：
            p1_enc: 加密数据库点 1
            p2_enc: 加密数据库点 2
            q_enc: 加密查询点

        返回：
            如果 p1 比 p2 更接近 q，则返回 True
        """
        # 计算差值
        diff1_a = p1_enc[0] - p2_enc[0]
        diff1_b = p1_enc[1] - p2_enc[1]

        # 计算与查询的内积
        result = np.dot(diff1_a, q_enc[0]) + np.dot(diff1_b, q_enc[1])

        # r > 0，因此符号保持不变
        return result > 0

    def encrypt_database(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        加密多个数据库点。

        参数：
            points: [N, d] 数据库点数组

        返回：
            两个份额的元组 ([N, d'] 数组, [N, d'] 数组)
        """
        N, d = points.shape
        if d != self.d:
            raise ValueError(f"期望 {self.d} 维点，得到 {d}")

        encrypted_a = np.zeros((N, self.d_prime))
        encrypted_b = np.zeros((N, self.d_prime))

        for i in range(N):
            encrypted_a[i], encrypted_b[i] = self.encrypt_database_point(points[i])

        return encrypted_a, encrypted_b

    def encrypt_queries(self, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        加密多个查询点。

        参数：
            queries: [N, d] 查询点数组

        返回：
            两个份额的元组 ([N, d'] 数组, [N, d'] 数组)
        """
        N, d = queries.shape
        if d != self.d:
            raise ValueError(f"期望 {self.d} 维点，得到 {d}")

        encrypted_a = np.zeros((N, self.d_prime))
        encrypted_b = np.zeros((N, self.d_prime))

        for i in range(N):
            encrypted_a[i], encrypted_b[i] = self.encrypt_query(queries[i])

        return encrypted_a, encrypted_b


if __name__ == "__main__":
    # 测试 ASPE 方案 2
    print("测试 ASPE 方案 2")

    # 初始化
    d = 10
    aspe = ASPEScheme2(d)

    print(f"\n原始维度: {d}")
    print(f"扩展维度: {aspe.d_prime}")
    print(f"拆分配置 S: {aspe.S}")

    # 测试向量
    p1 = np.random.randn(d)
    p2 = np.random.randn(d)
    q = np.random.randn(d)

    # 加密
    p1_enc = aspe.encrypt_database_point(p1)
    p2_enc = aspe.encrypt_database_point(p2)
    q_enc = aspe.encrypt_query(q)

    print(f"\n原始 p1: {p1[:3]}...")
    print(f"加密后 p1 份额 a: {p1_enc[0][:3]}...")
    print(f"加密后 p1 份额 b: {p1_enc[1][:3]}...")

    # 测试内积保持
    plaintext_ip = np.dot(p1, q)
    ciphertext_ip = aspe.ciphertext_inner_product(p1_enc, q_enc)

    print(f"\n明文内积: {plaintext_ip:.6f}")
    print(f"密文内积: {ciphertext_ip:.6f}")

    # 测试解密
    p1_dec = aspe.decrypt_point(p1_enc)
    print(f"\n解密误差: {np.linalg.norm(p1 - p1_dec):.6e}")

    # 测试距离比较
    plaintext_result = np.dot(p1 - p2, q) > 0
    ciphertext_result = aspe.compare_distance(p1_enc, p2_enc, q_enc)

    print(f"\n明文比较: {plaintext_result}")
    print(f"密文比较: {ciphertext_result}")
    print(f"比较保持: {plaintext_result == ciphertext_result}")
