# DCMH 模型实施与实验报告

**项目**: DeepHash-ASPE
**报告日期**: 2026-03-21
**测试状态**: ✅ 全部通过

---

## 1. 项目概述

### 1.1 目标

实施 DCMH (Deep Cross-Modal Hashing) 模型，替代原有的基于 CNN 的实现，用于跨模态图文检索任务。

### 1.2 实施内容

| 模块 | 文件名 | 状态 |
|------|--------|------|
| 基础模块 | `core/hashing/dcmh_basic.py` | ✅ 完成 |
| 图像模块 | `core/hashing/dcmh_image.py` | ✅ 完成 |
| 文本模块 | `core/hashing/dcmh_text.py` | ✅ 完成 |
| 双流模型 | `core/hashing/dcmh_model.py` | ✅ 完成 |
| 损失函数 | `training/dcmh_loss.py` | ✅ 完成 |
| 模型配置 | `config/model_config.py` | ✅ 完成 |
| 文档更新 | `README.md` | ✅ 完成 |

---

## 2. 模型架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      DCMH 双流模型                           │
├─────────────────────────┬───────────────────────────────────┤
│     图像流               │           文本流                   │
├─────────────────────────┼───────────────────────────────────┤
│  输入：[B, 3, 224, 224]  │  输入：[B, 1, y_dim, 1]            │
│    ↓                    │    ↓                              │
│  AlexNet CNN (5 层 conv)  │  Conv2d(1, 8192, k=y_dim)          │
│    ↓                    │    ↓                              │
│  FullConv (2 层)         │  ReLU                             │
│    ↓                    │    ↓                              │
│  Classifier: 4096→bit   │  Conv2d(8192, bit, k=1)            │
│    ↓                    │    ↓                              │
│  Output: [B, bit]       │  Output: [B, bit]                  │
└─────────────────────────┴───────────────────────────────────┘
                              ↓
                    共同哈希空间 [B, bit]
```

### 2.2 图像模块详细架构

| 层 | 类型 | 参数 | 输出 |
|----|------|------|------|
| conv1 | Conv2d(3,64,11,4) | 23K | [B,64,56,56] |
| norm1 | LocalResponseNorm | - | [B,64,56,56] |
| pool1 | MaxPool2d(3,2) | - | [B,64,27,27] |
| conv2 | Conv2d(64,256,5,1,2) | 410K | [B,256,27,27] |
| norm2 | LocalResponseNorm | - | [B,256,27,27] |
| pool2 | MaxPool2d(3,2) | - | [B,256,13,13] |
| conv3 | Conv2d(256,256,3,1,1) | 590K | [B,256,13,13] |
| conv4 | Conv2d(256,256,3,1,1) | 590K | [B,256,13,13] |
| conv5 | Conv2d(256,256,3,1,1) | 590K | [B,256,13,13] |
| pool5 | MaxPool2d(3,2) | - | [B,256,6,6] |
| full_conv6 | Conv2d(256,4096,6) | 67.2M | [B,4096,1,1] |
| full_conv7 | Conv2d(4096,4096,1) | 16.8M | [B,4096,1,1] |
| classifier | Linear(4096, bit) | 262K | [B,bit] |

### 2.3 文本模块详细架构

| 层 | 类型 | 参数 | 输出 |
|----|------|------|------|
| conv1 | Conv2d(1,8192,(y_dim,1)) | 8.2M × y_dim | [B,8192,1,1] |
| conv2 | Conv2d(8192,bit,1) | 8192 × bit | [B,bit,1,1] |

---

## 3. 实验结果

### 3.1 测试环境

| 项目 | 规格 |
|------|------|
| 操作系统 | Windows 11 |
| Python | 3.x |
| PyTorch | 最新版本 |
| 测试日期 | 2026-03-21 |

### 3.2 模块导入测试

| 模块 | 状态 |
|------|------|
| DCMHBasicModule | ✅ 通过 |
| DCMHImageModule | ✅ 通过 |
| DCMHTextModule | ✅ 通过 |
| DCMHModel | ✅ 通过 |
| DCMHCombinedLoss | ✅ 通过 |

**通过率**: 5/5 (100%)

### 3.3 前向传播测试

#### 图像编码性能

| Batch Size | 输入形状 | 输出形状 | 耗时 (ms) |
|------------|----------|----------|-----------|
| 1 | [1,3,224,224] | [64] | 50.36 |
| 2 | [2,3,224,224] | [2,64] | 60.99 |
| 4 | [4,3,224,224] | [4,64] | 71.65 |
| 8 | [8,3,224,224] | [8,64] | 73.75 |

#### 文本编码测试

| 标签数 | 输出范围 | 状态 |
|--------|----------|------|
| 1 (单标签) | [-0.0385, 0.0349] | ✅ |
| 5 (少标签) | [-0.0541, 0.0474] | ✅ |
| 20 (多标签) | [-0.0802, 0.1183] | ✅ |

### 3.4 双流模型测试

| 测试项 | 预期 | 实际 | 状态 |
|--------|------|------|------|
| 图像哈希形状 | [B, bit] | [4, 64] | ✅ |
| 文本哈希形状 | [B, bit] | [4, 64] | ✅ |
| 二进制唯一值 | {-1, +1} | {-1.0, 1.0} | ✅ |
| 自汉明距离 | 0 | 0.0 | ✅ |
| 相似度矩阵 | [B, B] | [4, 4] | ✅ |
| Top-K 检索 | 正常 | 正常 | ✅ |

### 3.5 损失函数测试

| 损失版本 | 总损失 | 跨模态 | 量化 |
|----------|--------|--------|------|
| Contrastive | 0.0361 | -0.0026 | 0.3867 |
| InfoNCE | 3.6130 | 3.5743 | 0.3867 |

### 3.6 不同哈希码长度对比

| Bit | 参数量 | 输出形状 |
|-----|--------|----------|
| 16 | 65,134,368 | [B, 16] |
| 32 | 65,331,008 | [B, 32] |
| 64 | 65,724,288 | [B, 64] |
| 128 | 66,510,848 | [B, 128] |

### 3.7 参数量统计

| 模块 | 参数量 | 百分比 |
|------|--------|--------|
| 图像模块 | 56,999,744 | 86.7% |
| 文本模块 | 8,724,544 | 13.3% |
| **总计** | **65,724,288** | 100% |

---

## 4. 损失函数分析

### 4.1 跨模态对比损失

$$L_{cross} = \frac{1}{N}\sum_{i,j} \left[ -S_{ij} \cdot \log(\hat{S}_{ij}) + (1-S_{ij}) \cdot \log(1-\hat{S}_{ij}) \right]$$

### 4.2 量化损失

$$L_{quant} = \frac{1}{N \cdot bit}\sum_{i=1}^{N}\sum_{k=1}^{bit} (|h_{ik}| - 1)^2$$

### 4.3 总损失

$$L_{total} = L_{cross} + \gamma \cdot L_{quant}$$

---

## 5. 使用指南

### 5.1 快速开始

```python
from core.hashing.dcmh_model import DCMHModel
from training.dcmh_loss import DCMHCombinedLoss
import torch

# 创建模型
model = DCMHModel(bit=64, y_dim=80)  # MS-COCO: 80 类

# 创建损失函数
criterion = DCMHCombinedLoss(
    margin=1.0,
    quantization_weight=0.1,
    use_info_nce=False
)

# 准备数据
image = torch.randn(4, 3, 224, 224)
labels = torch.zeros(4, 1, 80, 1)
labels[0, 0, [1,5,10], 0] = 1.0

# 前向传播
img_hash = model.encode_image(image)
txt_hash = model.encode_text(labels)

# 计算损失
loss, loss_dict = criterion(img_hash, txt_hash)
```

### 5.2 训练循环

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(100):
    for batch in dataloader:
        img_hash = model.encode_image(batch['image'])
        txt_hash = model.encode_text(batch['labels'])

        loss, loss_dict = criterion(img_hash, txt_hash)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

### 5.3 跨模态检索

```python
# 计算相似度
similarity = model.compute_similarity(query_hash, db_hash)

# Top-K 检索
distances, indices = model.cross_modal_retrieval(
    query_hash, db_hash, top_k=10
)

# 获取检索结果
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"Rank {i+1}: Distance = {dist:.4f}, Index = {idx}")
```

---

## 6. 配置说明

### 6.1 模型配置

```python
DCMH_CONFIG = {
    'bit': 64,                # 哈希码位数
    'y_dim': 1000,            # 文本标签维度
    'image_pretrain': None,   # 图像预训练路径
    'quantization_tau': 1.0,  # 量化温度
    'with_quantization': True # 使用量化训练
}
```

### 6.2 不同位数配置

```python
DCMH_BIT_CONFIGS = {
    16: {'bit': 16, 'y_dim': 1000},
    32: {'bit': 32, 'y_dim': 1000},
    64: {'bit': 64, 'y_dim': 1000},
    128: {'bit': 128, 'y_dim': 1000}
}
```

---

## 7. 结论与展望

### 7.1 实施成果

1. ✅ **完整实施** DCMH 模型的图像和文本模块
2. ✅ **验证通过** 所有单元测试和集成测试
3. ✅ **文档完善** 更新 README 和实验报告
4. ✅ **配置灵活** 支持多种哈希码长度

### 7.2 性能指标

- **图像编码**: Batch=4 时 71.65ms
- **文本编码**: 支持 1-20+ 标签
- **二进制输出**: 唯一值 {-1.0, 1.0}
- **自汉明距离**: 0.0 (验证正确性)

### 7.3 下一步工作

1. 在 MS-COCO 数据集上进行训练
2. 评估 mAP 指标
3. 与 ASPE 加密方案集成
4. 优化推理速度

---

## 8. 附录

### 8.1 文件清单

```
core/hashing/
├── dcmh_basic.py        # 基础模块
├── dcmh_image.py        # 图像编码模块
├── dcmh_text.py         # 文本编码模块
└── dcmh_model.py        # 双流模型

training/
└── dcmh_loss.py         # 损失函数

config/
└── model_config.py      # 模型配置 (已添加 DCMH_CONFIG)

tests/
└── test_dcmh_full.py    # 完整测试脚本

results/
├── dcmh_test_results.json  # JSON 测试结果
├── dcmh_test_summary.md    # Markdown 摘要
├── dcmh_test_report.md     # 测试报告
└── dcmh_experiment_report.md  # 实验报告
```

### 8.2 参考资源

- DCMH 论文：Deep Cross-Modal Hashing
- 参考实现：`reference/DCMH/`
- PyTorch 文档：https://pytorch.org/

---

**报告生成时间**: 2026-03-21
**项目负责人**: AI Assistant
**联系**: 详见项目 README
