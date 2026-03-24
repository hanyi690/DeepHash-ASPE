"""
ASPE 包装器：用于 DCMH 深度哈希跨模态检索

提供与 DCMH 代码风格一致的接口，支持：
- 哈希码加密（GenEnc/GenTrap）
- 密文汉明距离计算
- 密文 mAP 评估
"""

import numpy as np
import torch
from typing import Optional
from .scheme1 import ASPEScheme1


class ASPEForDCMH:
    """
    ASPE 加密与 DCMH 哈希码的接口类。

    将 DCMH 生成的 {-1, +1} 哈希码转换为 ASPE 可处理的形式，
    并在密文空间计算汉明距离和 mAP 指标。

    核心洞察：
    1. 汉明距离与内积的线性关系：hamm(B1, B2) = 0.5 × (bit - B1·B2)
    2. ASPE 内积保持性：EncDB(p)·EncQuery(q) = r × (p·q)，其中 r > 0
    3. 排序不变性：汉明距离排序 → 内积排序 → ASPE 密文内积排序
    """

    def __init__(self, bit_dim: int, seed: int = 42):
        """
        初始化 ASPE for DCMH。

        参数：
            bit_dim: 哈希码位数（如 16, 32, 64）
            seed: ASPE 密钥生成的随机种子
        """
        self.bit = bit_dim
        self.aspe = ASPEScheme1(d=bit_dim, seed=seed)

    def GenEnc(self, retrieval_codes: np.ndarray) -> np.ndarray:
        """
        加密检索库哈希码。

        将 DCMH 生成的 {-1, +1} 哈希码加密为 ASPE 密文。

        参数：
            retrieval_codes: {-1, +1}^{n×bit} 检索库哈希码（numpy 数组）

        返回：
            {n×(bit+1)} 加密检索库
        """
        if isinstance(retrieval_codes, torch.Tensor):
            retrieval_codes = retrieval_codes.cpu().numpy()

        # 确保是浮点类型（ASPE 需要）
        retrieval_float = retrieval_codes.astype(np.float64)

        # 使用 ASPE 加密
        return self.aspe.GenEnc(retrieval_float)

    def GenTrap(self, query_codes: np.ndarray, r: Optional[float] = None) -> np.ndarray:
        """
        加密查询哈希码（生成陷阱门）。

        参数：
            query_codes: {-1, +1}^{m×bit} 查询哈希码
            r: 可选的缩放因子（如果为 None，则使用固定值 1.0）

        返回：
            {m×(bit+1)} 加密查询（陷阱门）
        """
        if isinstance(query_codes, torch.Tensor):
            query_codes = query_codes.cpu().numpy()

        query_float = query_codes.astype(np.float64)
        return self.aspe.GenTrap(query_float, r)

    def ciphertext_hamming_distance(self, encrypted_q: np.ndarray,
                                    encrypted_r: np.ndarray) -> np.ndarray:
        """
        使用 ASPE 密文内积计算等效汉明距离。

        原理：
        对于 ASPE 方案 1 和 {-1, +1} 哈希码：
        - 密文内积：cipher_ip = p·q - 0.5×bit
        - 汉明距离：hamm = 0.5×(bit - p·q)
        - 因此：hamm = 0.25×bit - 0.5×cipher_ip

        参数：
            encrypted_q: 加密查询（单个或多个）
            encrypted_r: 加密检索库

        返回：
            汉明距离数组
        """
        if isinstance(encrypted_q, torch.Tensor):
            encrypted_q = encrypted_q.cpu().numpy()
        if isinstance(encrypted_r, torch.Tensor):
            encrypted_r = encrypted_r.cpu().numpy()

        # 确保是 2D 数组
        if encrypted_q.ndim == 1:
            encrypted_q = encrypted_q.reshape(1, -1)

        # 计算密文内积矩阵 [num_query, num_retrieval]
        inner_products = np.dot(encrypted_q, encrypted_r.T)

        # 转换回汉明距离
        # 对于 {-1, +1} 哈希码：hamm = 0.25×bit - 0.5×cipher_ip
        hamm = 0.25 * self.bit - 0.5 * inner_products

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
            encrypted_qB: 加密查询哈希码 [num_query, bit+1]
            encrypted_rB: 加密检索库哈希码 [num_retrieval, bit+1]
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
        min_overlap_ratios = {10: 0.9, 50: 0.85, 100: 0.80}

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


# 便捷函数：与 DCMH main.py 风格一致
def generate_encrypted_codes(model, data, bit_dim, aspe_wrapper):
    """
    生成加密哈希码（DCMH generate_*_code 的加密版本）。

    参数：
        model: DCMH 模型（ImgModule 或 TxtModule）
        data: 输入数据
        bit_dim: 哈希位数
        aspe_wrapper: ASPEForDCMH 实例

    返回：
        加密哈希码
    """
    # 生成原始哈希码
    B = generate_codes(model, data, bit_dim)

    # 加密
    return aspe_wrapper.GenEnc(B)


def generate_codes(model, Y, bit):
    """
    生成哈希码（参考 DCMH 的 generate_image_code / generate_text_code）。

    这是一个简化版本，实际使用时应传入正确的数据处理器。
    """
    # 这里假设模型可以直接处理输入并输出哈希码
    # 实际使用时需要根据具体模型调整
    import torch
    from tqdm import tqdm

    batch_size = 64  # 默认 batch size
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        B = B.cuda()
        model = model.cuda()

    with torch.no_grad():
        for i in tqdm(range(num_data // batch_size + 1)):
            ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]

            # 处理输入
            if isinstance(Y, torch.Tensor):
                data = Y[ind].type(torch.float)
            else:
                data = torch.from_numpy(Y[ind]).type(torch.float)

            # 文本数据需要额外的维度
            if data.dim() == 2:
                data = data.unsqueeze(1).unsqueeze(-1)

            if use_gpu:
                data = data.cuda()

            cur_f = model(data)
            B[ind, :] = cur_f.data

    B = torch.sign(B)
    return B
