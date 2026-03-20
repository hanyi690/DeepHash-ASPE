# DCMH 模型实验报告

## 1. 实验目的

本实验旨在实现和验证 DCMH (Deep Cross-Modal Hashing) 模型，用于跨模态图文检索任务。DCMH 是一种端到端的深度哈希模型，能够将图像和文本映射到共同的二进制哈希空间。

## 2. 模型架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      DCMH 模型架构                           │
├─────────────────────────┬───────────────────────────────────┤
│     图像流               │           文本流                   │
├─────────────────────────┼───────────────────────────────────┤
│  输入：[B, 3, 224, 224]  │  输入：[B, 1, y_dim, 1]            │
│    ↓                    │    ↓                              │
│  Conv1: 3→64, k=11, s=4  │  Conv1: 1→8192, k=(y_dim,1)        │
│    ↓                    │    ↓                              │
│  Norm + Pool            │  ReLU                             │
│    ↓                    │    ↓                              │
│  Conv2: 64→256, k=5     │  Conv2: 8192→bit, k=1              │
│    ↓                    │    ↓                              │
│  Conv3-5: 256→256       │  Output: [B, bit]                  │
│    ↓                    │                                   │
│  FullConv6: 256→4096    │                                   │
│    ↓                    │                                   │
│  FullConv7: 4096→4096   │                                   │
│    ↓                    │                                   │
│  Classifier: 4096→bit   │                                   │
│    ↓                    │                                   │
│  Output: [B, bit]       │                                   │
└─────────────────────────┴───────────────────────────────────┘
                              ↓
                    共同哈希空间 [B, bit]
```

### 2.2 图像模块 (DCMHImageModule)

| 层 | 类型 | 输入 | 输出 | 参数 |
|----|------|------|------|------|
| conv1 | Conv2d | [B,3,224,224] | [B,64,56,56] | 23K |
| norm1 | LocalResponseNorm | [B,64,56,56] | [B,64,56,56] | - |
| pool1 | MaxPool2d | [B,64,56,56] | [B,64,27,27] | - |
| conv2 | Conv2d | [B,64,27,27] | [B,256,27,27] | 410K |
| norm2 | LocalResponseNorm | [B,256,27,27] | [B,256,27,27] | - |
| pool2 | MaxPool2d | [B,256,27,27] | [B,256,13,13] | - |
| conv3 | Conv2d | [B,256,13,13] | [B,256,13,13] | 590K |
| conv4 | Conv2d | [B,256,13,13] | [B,256,13,13] | 590K |
| conv5 | Conv2d | [B,256,13,13] | [B,256,13,13] | 590K |
| pool5 | MaxPool2d | [B,256,13,13] | [B,256,6,6] | - |
| full_conv6 | Conv2d | [B,256,6,6] | [B,4096,1,1] | 67.2M |
| full_conv7 | Conv2d | [B,4096,1,1] | [B,4096,1,1] | 16.8M |
| classifier | Linear | [B,4096] | [B,bit] | 262K |

**总参数**: 约 86.5M

### 2.3 文本模块 (DCMHTextModule)

| 层 | 类型 | 输入 | 输出 | 参数 |
|----|------|------|------|------|
| conv1 | Conv2d | [B,1,y_dim,1] | [B,8192,1,1] | 8.2M × y_dim |
| conv2 | Conv2d | [B,8192,1,1] | [B,bit,1,1] | 8192 × bit |

**总参数**: 约 8.2M (y_dim=1000, bit=64)

## 3. 损失函数

### 3.1 跨模态对比损失

$$L_{cross} = \frac{1}{N}\sum_{i,j} \left[ -S_{ij} \cdot \log(\hat{S}_{ij}) + (1-S_{ij}) \cdot \log(1-\hat{S}_{ij}) \right]$$

其中 $S_{ij}$ 是样本 i 和 j 的相似度标签，$\hat{S}_{ij}$ 是预测的相似度。

### 3.2 量化损失

$$L_{quant} = \frac{1}{N \cdot bit}\sum_{i=1}^{N}\sum_{k=1}^{bit} (|h_{ik}| - 1)^2$$

鼓励哈希码接近二进制值 {-1, +1}。

### 3.3 总损失

$$L_{total} = L_{cross} + \gamma \cdot L_{quant}$$

其中 $\gamma$ 是量化损失的权重（默认 0.1）。

## 4. 实验设置

### 4.1 硬件环境

| 组件 | 规格 |
|------|------|
| CPU | Intel Core i7 / AMD Ryzen 7 |
| GPU | NVIDIA RTX 3080 / 4090 |
| 内存 | 32GB DDR4 |
| 存储 | NVMe SSD |

### 4.2 软件环境

| 软件 | 版本 |
|------|------|
| Python | 3.8+ |
| PyTorch | 1.9+ |
| CUDA | 11.0+ |
| OS | Windows 11 / Linux |

### 4.3 超参数设置

| 参数 | 值 |
|------|-----|
| batch_size | 64 |
| learning_rate | 1e-4 |
| bit | 16/32/64/128 |
| y_dim | 1000 (MS-COCO: 80) |
| margin | 1.0 |
| quantization_weight | 0.1 |
| epochs | 100 |

## 5. 实验结果

### 5.1 模型输出验证

| 测试项 | 预期输出 | 实际输出 | 状态 |
|--------|----------|----------|------|
| 图像编码形状 | [B, bit] | [4, 64] | ✅ |
| 文本编码形状 | [B, bit] | [4, 64] | ✅ |
| 二进制值 | {-1, +1} | {-1.0, 1.0} | ✅ |
| 自汉明距离 | 0 | 0.0 | ✅ |
| 跨模态相似度 | [B, B] | [4, 4] | ✅ |

### 5.2 参数量统计

| 模块 | 参数量 |
|------|--------|
| DCMHImageModule | ~65.7M |
| DCMHTextModule | ~8.2M |
| DCMHModel (总计) | ~65.7M |

### 5.3 损失函数验证

| 损失类型 | 初始值 | 说明 |
|----------|--------|------|
| 跨模态损失 | 0.0043 | 对比损失 |
| 量化损失 | 0.4006 | 鼓励二值化 |
| 总损失 | 0.0444 | 加权和 |

### 5.4 InfoNCE 损失对比

| 损失类型 | 值 | 特点 |
|----------|------|------|
| Contrastive | 0.0444 | 基于边界 |
| InfoNCE | 3.3344 | 基于温度 |

## 6. 模型特性分析

### 6.1 优势

1. **端到端训练**: 无需分阶段训练，直接优化哈希码
2. **跨模态对齐**: 图像和文本直接映射到共同哈希空间
3. **量化感知**: 内置量化损失，提高二值化质量
4. **高效检索**: 二进制哈希码支持快速汉明距离计算

### 6.2 局限

1. **参数量大**: 图像模块约 65M 参数
2. **标签依赖**: 文本模块需要预定义的标签维度
3. **固定输入**: 图像尺寸固定为 224×224

### 6.3 改进方向

1. **轻量化**: 使用 MobileNet 等轻量骨干
2. **动态标签**: 支持变长文本输入
3. **多标签增强**: 扩展标签语义表示

## 7. 使用指南

### 7.1 模型加载

```python
from core.hashing.dcmh_model import DCMHModel

# 创建模型
model = DCMHModel(bit=64, y_dim=80)  # MS-COCO: 80 类

# 加载预训练权重
model.load_state_dict(torch.load('checkpoints/dcmh_best.pth'))
```

### 7.2 特征提取

```python
# 图像编码
img_hash = model.encode_image(image_tensor)

# 文本编码
txt_hash = model.encode_text(text_labels)

# 二进制哈希码
binary_img = torch.sign(img_hash)
binary_txt = torch.sign(txt_hash)
```

### 7.3 跨模态检索

```python
# 计算相似度
similarity = model.compute_similarity(query_hash, db_hash)

# Top-K 检索
distances, indices = model.cross_modal_retrieval(
    query_hash, db_hash, top_k=10
)
```

### 7.4 训练流程

```python
from training.dcmh_loss import DCMHCombinedLoss
import torch.optim as optim

# 初始化
model = DCMHModel(bit=64, y_dim=80)
criterion = DCMHCombinedLoss(margin=1.0, quantization_weight=0.1)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练循环
for epoch in range(100):
    for batch in dataloader:
        img_hash = model.encode_image(batch['image'])
        txt_hash = model.encode_text(batch['labels'])

        loss, loss_dict = criterion(img_hash, txt_hash)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 8. 结论

本实验成功实现并验证了 DCMH 模型的核心功能：

1. ✅ **图像编码模块**: AlexNet 风格 CNN，输出 64 位哈希码
2. ✅ **文本编码模块**: 双层卷积，支持 multi-hot 标签输入
3. ✅ **双流整合**: 统一的跨模态哈希空间
4. ✅ **损失函数**: 跨模态对比损失 + 量化损失
5. ✅ **检索功能**: 支持跨模态 Top-K 检索

模型已准备就绪，可进行下一步的 MS-COCO 数据集训练和评估。

---

**报告生成日期**: 2026-03-21
**实验状态**: ✅ 完成
**下一步**: MS-COCO 数据集训练
