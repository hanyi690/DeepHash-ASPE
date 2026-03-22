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
    计算 mAP@K。

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
    map_score = 0
    if k is None:
        k = retrieval_L.shape[0]

    for iter_idx in range(num_query):
        q_L = query_L[iter_idx]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter_idx, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
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
