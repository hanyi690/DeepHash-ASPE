"""
深度哈希模型配置

双流深度哈希模型的配置，该模型将图像和文本映射到共同的嵌入空间。
"""

# 双流哈希模型配置
DUAL_STREAM_CONFIG = {
    'feature_dim': 4096,      # 共同嵌入空间维度
    'hash_bits': 64,          # 可选的二进制哈希码长度
    'dropout': 0.5,

    # 图像编码器
    'image': {
        'backbone': 'resnet50',  # 或 'vgg16', 'efficientnet'
        'pretrained': True,
        'freeze_backbone': False,
        'input_size': 224
    },

    # 文本编码器
    'text': {
        'vocab_size': 10000,    # 将基于 MS-COCO 词汇表设置
        'embed_dim': 300,
        'num_layers': 6,
        'num_heads': 8,
        'max_seq_length': 50,
        'use_pretrained': True   # 使用预训练词嵌入
    }
}

# 图像特定配置
IMAGE_ENCODER_CONFIG = {
    'backbone': 'resnet50',
    'feature_dim': 4096,
    'hash_bits': 64,
    'pretrained': True,
    'freeze_backbone': False
}

# 文本特定配置
TEXT_ENCODER_CONFIG = {
    'vocab_size': 10000,
    'feature_dim': 4096,
    'hash_bits': 64,
    'embed_dim': 300,
    'num_layers': 6,
    'num_heads': 8,
    'max_seq_length': 50,
    'dropout': 0.1
}

# DCMH (Deep Cross-Modal Hashing) 模型配置
DCMH_CONFIG = {
    'bit': 64,                # 哈希码位数 (16/32/64/128)
    'y_dim': 1000,            # 文本标签维度（MS-COCO 为 80 类，可扩展）
    'image_pretrain': None,   # 图像模块预训练路径（可选）
    'quantization_tau': 1.0,  # 量化温度参数
    'with_quantization': True # 是否使用量化训练
}

# DCMH 不同哈希码长度配置
DCMH_BIT_CONFIGS = {
    16: {'bit': 16, 'y_dim': 1000},
    32: {'bit': 32, 'y_dim': 1000},
    64: {'bit': 64, 'y_dim': 1000},
    128: {'bit': 128, 'y_dim': 1000}
}
