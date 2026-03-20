"""
训练配置

用于训练双流深度哈希模型的配置。
"""

TRAINING_CONFIG = {
    # 基础训练参数
    'num_epochs': 100,
    'batch_size': 64,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,

    # 学习率调度
    'lr_scheduler': 'step',  # 'step', 'cosine', 'plateau'
    'lr_decay_rate': 0.1,
    'lr_decay_epochs': [30, 60, 80],

    # 损失函数
    'loss_type': 'cross_modal',  # 'contrastive', 'triplet', 'cross_modal'
    'margin': 1.0,
    'alpha': 0.5,  # 模态内损失和模态间损失的平衡系数

    # 优化器
    'optimizer': 'adam',
    'momentum': 0.9,
    'early_stopping': True,
    'patience': 10,

    # 检查点保存
    'checkpoint_dir': './checkpoints',
    'save_frequency': 5,
    'resume_from': None
}

# 验证配置
VALIDATION_CONFIG = {
    'validate_every': 1,  # 每隔 N 轮验证一次
    'save_best_only': True,
    'metric': 'map'  # 监控此指标以保存最佳模型
}
