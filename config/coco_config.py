"""
MS-COCO 数据集配置

用于加载和预处理 MS-COCO 2014 数据集的配置，
用于训练和评估。
"""
import os

# MS-COCO 2014 数据集
COCO_CONFIG = {
    # 数据路径
    'data_root': './data/coco',
    'images_dir': './data/coco/images',
    'annotations_file': './data/coco/annotations/captions_train2014.json',
    'val_annotations': './data/coco/annotations/captions_val2014.json',

    # 数据集划分
    'train_split': 'train2014',
    'val_split': 'val2014',
    'test_split': 'val2014',  # MS-COCO 使用验证集进行测试

    # 数据处理
    'image_size': 224,
    'max_caption_length': 50,
    'vocab_threshold': 5,     # 词汇表中最小词频

    # 训练设置
    'batch_size': 64,
    'num_workers': 4,

    # 跨模态检索设置
    'num_captions_per_image': 5,  # MS-COCO 每张图像有5个标题
    'use_all_captions': True
}

# 文本预处理配置
TEXT_PREPROCESSING_CONFIG = {
    'tokenizer': 'nltk',  # 或 'spacy', 'bert'
    'lowercase': True,
    'remove_punctuation': True,
    'stemming': False,
    'lemmatization': False
}
