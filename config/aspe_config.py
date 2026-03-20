"""
ASPE 配置

ASPE 方案1（2级安全）和 ASPE 方案2（3级安全）的配置。
"""

# ASPE 方案1：基础2级安全
ASPE_SCHEME1_CONFIG = {
    'feature_dim': 4096,  # 从哈希模型提取的特征向量维度
    'security_level': 2,
    'matrix_size': 4097   # 方案1的 (d+1) x (d+1) 矩阵
}

# ASPE 方案2：增强3级安全
ASPE_SCHEME2_CONFIG = {
    'feature_dim': 4096,
    'security_level': 3,
    'd_prime': max(4097, 80),  # 带有人工属性的扩展维度
    'artificial_dimensions': 80,
    'splitting_config': None  # 作为位向量随机生成
}

# 用于可重复性的随机种子
RANDOM_SEED = 42
