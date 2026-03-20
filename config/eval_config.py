"""
评估配置

用于评估隐私保护检索系统的配置。
"""

EVALUATION_CONFIG = {
    # 检索设置
    'k_values': [1, 5, 10, 20, 50],  # 用于 Precision@K 和 Recall@K
    'num_queries': 1000,             # 测试查询数量

    # 评估用的 ASPE 方案
    'aspe_scheme': 'scheme1',        # 'scheme1' 或 'scheme2'

    # 要计算的指标
    'metrics': ['map', 'precision@k', 'recall@k', 'ndcg'],

    # 性能基准测试
    'measure_time': True,
    'compare_unencrypted': True,     # 与未加密检索进行对比

    # 安全性评估
    'security_tests': True,
    'attack_samples': 100            # 用于攻击测试的样本数量
}

# 检索模式配置
RETRIEVAL_MODES = {
    'text_to_image': {
        'query_modality': 'text',
        'target_modality': 'image'
    },
    'image_to_text': {
        'query_modality': 'image',
        'target_modality': 'text'
    },
    'image_to_image': {
        'query_modality': 'image',
        'target_modality': 'image'
    },
    'text_to_text': {
        'query_modality': 'text',
        'target_modality': 'text'
    }
}
