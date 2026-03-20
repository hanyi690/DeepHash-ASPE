"""
ASPE 方案 1：基础 2 级安全

实现基础非对称标量积保持加密（ASPE）
具有 2 级安全性（抵抗已知样本攻击）。

核心思想：将 d 维向量扩展到 (d+1) 维，并使用可逆矩阵 M 进行变换
以保持标量积。
"""

import numpy as np
from typing import Optional
from core.aspe.keygen import generate_key_scheme1


class ASPEScheme1:
    """
    用于隐私保护相似度搜索的 ASPE 方案 1。

    安全级别：2（抵抗已知样本攻击）

    加密策略：
    - 数据库点 p：扩展为 p̂ = (p, -0.5||p||²)，使用 M^T 变换
    - 查询点 q：扩展为 q̂ = r*(q, 1)，使用 M^(-1) 变换
    - 保持内积：p̂' · q̂' = r * (p · q)

    属性：
    - 距离不可恢复：无法恢复精确的欧几里得距离
    - 陷阱门不可链接：无法将陷阱门与查询关联
    """

    def __init__(self, d: int, seed: int = 42):
        """
        初始化 ASPE 方案 1。

        参数：
            d: 特征维度
            seed: 密钥生成的随机种子
        """
        self.d = d
        self.seed = seed

        # 生成密钥矩阵 M
        self.M = generate_key_scheme1(d, seed)

        # 预计算逆矩阵和转置矩阵
        try:
            self.M_inv = np.linalg.inv(self.M)
            self.M_T = self.M.T
            self.M_inv_T = self.M_inv.T
        except np.linalg.LinAlgError:
            raise ValueError("密钥矩阵求逆失败。使用正确的密钥生成方法不应发生此错误。")

    def encrypt_database_point(self, p: np.ndarray) -> np.ndarray:
        """
        使用方案 1 加密数据库点 p。

        步骤：
        1. 将 p 扩展到 (d+1) 维：p̂ = (p, -0.5||p||²)
        2. 变换：p' = M^T * p̂

        参数：
            p: d 维数据库点

        返回：
            (d+1) 维加密点
        """
        if p.shape[0] != self.d:
            raise ValueError(f"期望 {self.d} 维点，得到 {p.shape[0]}")

        # 步骤 1：将点扩展到 (d+1) 维
        p_hat = np.zeros(self.d + 1)
        p_hat[:self.d] = p
        p_hat[self.d] = -0.5 * np.dot(p, p)  # -0.5||p||²

        # 步骤 2：使用 M^T 变换
        p_enc = np.dot(self.M_T, p_hat)

        return p_enc

    def encrypt_query(self, q: np.ndarray, r: Optional[float] = None) -> np.ndarray:
        """
        使用方案 1 加密查询点 q。

        步骤：
        1. 生成随机缩放因子 r > 0（如果未提供）
        2. 将 q 扩展到 (d+1) 维：q̂ = r * (q, 1)
        3. 变换：q' = M^(-1) * q̂

        参数：
            q: d 维查询点
            r: 可选的随机缩放因子（如果未提供则生成）

        返回：
            (d+1) 维加密查询（陷阱门）
        """
        if q.shape[0] != self.d:
            raise ValueError(f"期望 {self.d} 维点，得到 {q.shape[0]}")

        # 步骤 1：生成随机缩放因子
        if r is None:
            r = np.random.uniform(0.1, 10.0)
        elif r <= 0:
            raise ValueError("缩放因子 r 必须为正数")

        # 步骤 2：将查询扩展到 (d+1) 维
        q_hat = np.zeros(self.d + 1)
        q_hat[:self.d] = r * q
        q_hat[self.d] = r

        # 步骤 3：使用 M^(-1) 变换
        q_enc = np.dot(self.M_inv, q_hat)

        return q_enc

    def decrypt_point(self, p_enc: np.ndarray) -> np.ndarray:
        """
        解密加密点。

        步骤：
        1. 应用 M^(-T) 获取扩展点：p̂ = M^(-T) * p'
        2. 提取前 d 维（丢弃 -0.5||p||² 分量）

        参数：
            p_enc: (d+1) 维加密点

        返回：
            d 维解密点
        """
        if p_enc.shape[0] != self.d + 1:
            raise ValueError(f"期望 {self.d+1} 维加密点，得到 {p_enc.shape[0]}")

        # 步骤 1：逆变换
        p_hat = np.dot(self.M_inv_T, p_enc)

        # 步骤 2：提取前 d 维
        p = p_hat[:self.d]

        return p

    def ciphertext_inner_product(self,
                                 p_enc: np.ndarray,
                                 q_enc: np.ndarray) -> float:
        """
        在密文空间计算内积。

        对于方案 1：
        p_enc · q_enc = p · q（保持 2 型标量积）

        这允许在无需解密的情况下进行相似度比较。

        参数：
            p_enc: 加密数据库点
            q_enc: 加密查询点

        返回：
            内积值（与相似度成正比）
        """
        if p_enc.shape != q_enc.shape:
            raise ValueError(f"形状不匹配：{p_enc.shape} vs {q_enc.shape}")

        # 加密空间中的内积等于 r * (p · q)
        # 其中 r 是查询加密中使用的随机缩放因子
        return np.dot(p_enc, q_enc)

    def compare_distance(self,
                        p1_enc: np.ndarray,
                        p2_enc: np.ndarray,
                        q_enc: np.ndarray) -> bool:
        """
        比较距离：检查 (p1 - p2) · q > 0

        在密文空间：
        (p1' - p2') · q' = r * ((p1 - p2) · q)

        由于 r > 0，这保持了比较结果。

        参数：
            p1_enc: 加密数据库点 1
            p2_enc: 加密数据库点 2
            q_enc: 加密查询点

        返回：
            如果 p1 比 p2 更接近 q，则返回 True
        """
        # 计算差值
        diff = p1_enc - p2_enc

        # 检查与查询的内积
        result = np.dot(diff, q_enc)

        # r > 0，因此符号保持不变
        return result > 0

    def encrypt_database(self, points: np.ndarray) -> np.ndarray:
        """
        加密多个数据库点。

        参数：
            points: [N, d] 数据库点数组

        返回：
            [N, d+1] 加密点数组
        """
        N, d = points.shape
        if d != self.d:
            raise ValueError(f"期望 {self.d} 维点，得到 {d}")

        encrypted = np.zeros((N, self.d + 1))
        for i in range(N):
            encrypted[i] = self.encrypt_database_point(points[i])

        return encrypted

    def encrypt_queries(self, queries: np.ndarray, r: Optional[float] = None) -> np.ndarray:
        """
        加密多个查询点。

        参数：
            queries: [N, d] 查询点数组
            r: 可选的缩放因子（所有查询相同）。如果为 None，则使用固定的默认值 1.0

        返回：
            [N, d+1] 加密查询数组
        """
        N, d = queries.shape
        if d != self.d:
            raise ValueError(f"期望 {self.d} 维点，得到 {d}")

        # 如果未提供 r，使用固定值 1.0 以确保多个查询之间的可比性
        if r is None:
            r = 1.0

        encrypted = np.zeros((N, self.d + 1))
        for i in range(N):
            encrypted[i] = self.encrypt_query(queries[i], r)

        return encrypted

    def GenEnc(self, database_vectors: np.ndarray) -> np.ndarray:
        """
        加密检索库（数据库）向量 - DCMH 风格接口。

        这是 encrypt_database 的别名，为了与 DCMH 代码风格保持一致。

        参数：
            database_vectors: [N, d] 检索库向量数组，每个向量应为 {-1, +1} 或任意实数值

        返回：
            [N, d+1] 加密向量数组
        """
        return self.encrypt_database(database_vectors)

    def GenTrap(self, query_vectors: np.ndarray, r: Optional[float] = None) -> np.ndarray:
        """
        生成查询陷阱门 - DCMH 风格接口。

        这是 encrypt_query 的批量版本，为了与 DCMH 代码风格保持一致。

        参数：
            query_vectors: [N, d] 查询向量数组
            r: 可选的缩放因子（如果为 None，则使用固定值 1.0 以保证多个查询之间的可比性）

        返回：
            [N, d+1] 加密查询（陷阱门）数组
        """
        return self.encrypt_queries(query_vectors, r)


if __name__ == "__main__":
    # 测试 ASPE 方案 1
    print("测试 ASPE 方案 1")

    # 初始化
    d = 10
    aspe = ASPEScheme1(d)

    # 测试向量
    p1 = np.random.randn(d)
    p2 = np.random.randn(d)
    q = np.random.randn(d)

    # 加密
    p1_enc = aspe.encrypt_database_point(p1)
    p2_enc = aspe.encrypt_database_point(p2)
    q_enc = aspe.encrypt_query(q)

    print(f"\n原始 p1: {p1[:3]}...")
    print(f"加密后 p1: {p1_enc[:3]}...")

    # 测试内积保持
    plaintext_ip = np.dot(p1, q)
    ciphertext_ip = aspe.ciphertext_inner_product(p1_enc, q_enc)

    print(f"\n明文内积: {plaintext_ip:.6f}")
    print(f"密文内积: {ciphertext_ip:.6f}")
    print(f"比率（应该是 r）: {ciphertext_ip / plaintext_ip:.6f}")

    # 测试解密
    p1_dec = aspe.decrypt_point(p1_enc)
    print(f"\n解密误差: {np.linalg.norm(p1 - p1_dec):.6e}")

    # 测试距离比较
    plaintext_result = np.dot(p1 - p2, q) > 0
    ciphertext_result = aspe.compare_distance(p1_enc, p2_enc, q_enc)

    print(f"\n明文比较: {plaintext_result}")
    print(f"密文比较: {ciphertext_result}")
    print(f"比较保持: {plaintext_result == ciphertext_result}")
