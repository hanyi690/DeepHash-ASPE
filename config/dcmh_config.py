"""
DCMH 配置

提供 DCMH 模型的默认配置。
"""

import warnings


class DCMHConfig(object):
    """
    DCMH 默认配置类。
    """

    # 模型路径
    load_img_path = None  # 加载图像模型路径
    load_txt_path = None  # 加载文本模型路径

    # 数据参数
    data_path = './data/FLICKR-25K.mat'
    pretrain_model_path = './data/imagenet-vgg-f.mat'
    training_size = 10000
    query_size = 2000
    database_size = 18015
    batch_size = 128
    num_workers = 0  # DataLoader 工作进程数

    # 超参数
    max_epoch = 500
    gamma = 1
    eta = 1
    bit = 64  # 最终二进制码长度
    lr = 10 ** (-1.5)  # 初始学习率 ≈ 0.0316，与 reference/DCMH 一致

    # 设备参数
    use_gpu = True

    # 验证间隔（每 N 个 epoch 验证一次，0 表示禁用训练中验证）
    valid_interval = 10  # 默认每 10 个 epoch 验证一次

    # 打印频率
    print_freq = 2  # 每 N 个 epoch 打印一次

    # 结果目录
    result_dir = 'result'

    # 断点恢复配置
    checkpoint_interval = 10  # 每 N 个 epoch 保存一次检查点
    resume_from = None  # 从指定检查点恢复（目录路径）

    def parse(self, kwargs):
        """
        通过 kwargs 更新配置。

        参数：
            kwargs: 配置参数字典
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('User config:')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


# 默认配置实例
opt = DCMHConfig()
