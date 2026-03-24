"""
评估模块

提供 DCMH 模型的评估功能：
- 检索指标计算（mAP, Precision@K, Recall@K）
- 哈希码质量评估
- GPU 加速的 mAP 计算
- 可视化图表生成
- 统一评估器

使用示例：
    from evaluation import DCMHEvaluator, calc_map_k

    # 使用统一评估器
    evaluator = DCMHEvaluator(result_dir='results/flickr-25k')
    results = evaluator.evaluate()

    # 直接使用 mAP 计算函数
    import torch
    qB = torch.randn(100, 64).sign()
    rB = torch.randn(1000, 64).sign()
    query_L = torch.randint(0, 2, (100, 10)).float()
    retrieval_L = torch.randint(0, 2, (1000, 10)).float()
    map_score = calc_map_k(qB, rB, query_L, retrieval_L)
"""

# 核心指标函数
from .metrics import (
    # 基础指标
    compute_average_precision,
    compute_map,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_ndcg_at_k,
    compute_retrieval_metrics,
    evaluate_cross_modal_retrieval,
    compute_hash_quality,
    # GPU 加速指标
    calc_hamming_dist,
    calc_map_k,
)

# 可视化函数
from .visualization import (
    plot_map_comparison,
    plot_precision_recall,
    plot_hash_quality_radar,
    plot_training_curves,
    plot_aspe_comparison,
    generate_evaluation_report,
)

# 评估器
from .evaluator import DCMHEvaluator

__all__ = [
    # 指标
    'compute_average_precision',
    'compute_map',
    'compute_precision_at_k',
    'compute_recall_at_k',
    'compute_ndcg_at_k',
    'compute_retrieval_metrics',
    'evaluate_cross_modal_retrieval',
    'compute_hash_quality',
    'calc_hamming_dist',
    'calc_map_k',
    # 可视化
    'plot_map_comparison',
    'plot_precision_recall',
    'plot_hash_quality_radar',
    'plot_training_curves',
    'plot_aspe_comparison',
    'generate_evaluation_report',
    # 评估器
    'DCMHEvaluator',
]