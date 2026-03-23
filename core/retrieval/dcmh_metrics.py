"""
DCMH 工具函数

提供 mAP 计算、汉明距离计算等评估函数。
"""

import torch


def calc_hammingDist(B1, B2):
    """
    计算汉明距离。

    参数：
        B1: 哈希码 [m, q]
        B2: 哈希码 [n, q]

    返回：
        汉明距离矩阵 [m, n]
    """
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    """
    计算 mAP@K（GPU 加速版本）。

    优化策略：
    1. 向量化计算所有汉明距离矩阵（一次性）
    2. 向量化计算所有 ground truth 矩阵（一次性）
    3. 保持张量在 GPU 上，减少数据传输

    参数：
        qB: {-1,+1}^{mxq} 查询哈希码
        rB: {-1,+1}^{nxq} 检索库哈希码
        query_L: {0,1}^{mxl} 查询标签
        retrieval_L: {0,1}^{nxl} 检索库标签
        k: 可选的截断位置

    返回：
        mAP 值
    """
    num_query = query_L.shape[0]
    num_retrieval = retrieval_L.shape[0]
    if k is None:
        k = num_retrieval

    # 向量化：一次性计算所有汉明距离 [num_query, num_retrieval]
    hamm = calc_hammingDist(qB, rB)

    # 向量化：一次性计算所有 ground truth [num_query, num_retrieval]
    gnd = (query_L.mm(retrieval_L.transpose(0, 1)) > 0).float()

    # 计算 mAP
    map_score = 0.0
    for i in range(num_query):
        gnd_i = gnd[i]
        tsum = gnd_i.sum()
        if tsum == 0:
            continue

        # 按汉明距离排序
        _, ind = torch.sort(hamm[i])
        gnd_i = gnd_i[ind]

        total = min(k, int(tsum))
        count = torch.arange(1, total + 1, dtype=torch.float32)
        if gnd_i.is_cuda:
            count = count.cuda()

        tindex = torch.nonzero(gnd_i)[:total].squeeze().float() + 1.0
        map_score = map_score + torch.mean(count / tindex)

    map_score = map_score / num_query
    return map_score


if __name__ == '__main__':
    # 测试 calc_map_k
    qB = torch.Tensor([[1, -1, 1, 1],
                       [-1, -1, -1, 1],
                       [1, 1, -1, 1],
                       [1, 1, 1, -1]])
    rB = torch.Tensor([[1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [1, 1, -1, -1],
                       [-1, 1, -1, -1],
                       [1, 1, -1, 1]])
    query_L = torch.Tensor([[0, 1, 0, 0],
                            [1, 1, 0, 0],
                            [1, 0, 0, 1],
                            [0, 1, 0, 1]])
    retrieval_L = torch.Tensor([[1, 0, 0, 1],
                                [1, 1, 0, 0],
                                [0, 1, 1, 0],
                                [0, 0, 1, 0],
                                [1, 0, 0, 0],
                                [0, 0, 1, 0]])

    map_result = calc_map_k(qB, rB, query_L, retrieval_L)
    print(f"mAP: {map_result}")
