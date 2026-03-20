# DeepHash-ASPE：隐私保护跨模态检索系统

一个结合了深度学习哈希与非对称标量积保持加密（ASPE）的隐私保护图文检索系统。

## 📢 更新 (2026-03-21)

### DCMH + ASPE 综合测试报告生成

已完成 **DCMH + ASPE 综合测试**，生成完整的测试报告和可视化图表：

- 📊 **6 张可视化图表**：参数量、编码时间、mAP 保持性、跨模态检索、哈希质量、加密开销
- 📝 **综合测试报告**：包含所有测试数据和结果分析
- ✅ **100% 通过率**：12/12 mAP 保持性测试通过
- 🔒 **安全属性验证**：陷阱门不可链接、距离比较保持

详见 [results/dcmh_aspe_comprehensive_report.md](results/dcmh_aspe_comprehensive_report.md)

### DCMH 模型架构实施完成

已完成 **DCMH (Deep Cross-Modal Hashing)** 模型的完整实施，替代了原有的基于 CNN 的实现：

- ✅ **DCMHImageModule**：AlexNet 风格的 CNN 图像编码器
- ✅ **DCMHTextModule**：基于卷积的文本标签编码器
- ✅ **DCMHModel**：统一的双流哈希模型
- ✅ **DCMHCombinedLoss**：跨模态对比损失 + 量化损失

详见 [DCMH 模型架构](#dcmh 模型架构)

### DCMH + ASPE 演示系统

新增 **DCMH + ASPE 演示系统** - 基于 Next.js 14 和 FastAPI 的完整前后端演示系统！

- 🌐 **Next.js 14 前端**：5 个交互式页面
- ⚡ **FastAPI 后端**：完整的 REST API
- 🔐 **隐私保护检索**：完整的 ASPE 加密流程
- 📊 **mAP 可视化**：明文/密文性能对比

详见 [演示系统文档](#dcmh--aspe 演示系统)

---

## 系统概述

本系统支持：
- **跨模态检索**：文本 ↔ 图像搜索
- **隐私保护**：使用 ASPE 加密实现加密数据库检索
- **全检索模式**：文本→图像、图像→文本、图像→图像、文本→文本
- **双重安全级别**：方案 1（2 级安全）和方案 2（3 级安全）
- **DCMH 集成**：支持深度跨模态哈希（DCMH）模型的加密输出

### 系统工作流程

```
1. 特征提取（深度哈希模型）
   ├─ 图像 → CNN (DCMHImageModule) → 特征向量
   └─ 文本 → 卷积标签处理 (DCMHTextModule) → 特征向量
   │
   └─ 输出：{-1, +1} 二进制哈希码

2. ASPE 加密
   ├─ 数据库哈希码 → GenEnc() → 加密检索库
   └─ 查询哈希码 → GenTrap() → 陷阱门

3. 隐私保护检索
   ├─ 计算密文汉明距离（等价于明文排序）
   └─ 按相似度排序 → 返回结果

注：DCMH (Deep Cross-Modal Hashing) 使用 AlexNet 风格 CNN 处理图像，
    使用两层卷积处理文本标签，直接输出哈希码。
```

## 项目结构

```
DeepHash-ASPE/
├── config/              # 配置文件
│   ├── model_config.py          # 模型配置（包含 DCMH_CONFIG）
│   ├── aspe_config.py           # ASPE 加密配置
│   ├── train_config.py          # 训练参数配置
│   └── coco_config.py           # MS-COCO 数据集配置
├── core/
│   ├── aspe/           # ASPE 加密方案
│   │   ├── scheme1.py           # 基础 2 级安全方案
│   │   ├── scheme2.py           # 增强 3 级安全方案
│   │   ├── keygen.py            # 密钥生成
│   │   └── dcmh_wrapper.py      # DCMH 集成包装器
│   ├── hashing/        # 深度哈希模型
│   │   ├── dcmh_basic.py        # DCMH 基础模块
│   │   ├── dcmh_image.py        # DCMH 图像编码模块
│   │   ├── dcmh_text.py         # DCMH 文本编码模块
│   │   ├── dcmh_model.py        # DCMH 双流模型
│   │   ├── image_hash.py        # 基于 CNN 的图像模型（备选）
│   │   ├── text_hash.py         # 基于 NLP 的文本模型（备选）
│   │   └── dual_stream.py       # 双流模型（旧版，保留兼容）
│   └── retrieval/      # 隐私保护检索
├── reference/
│   └── DCMH/           # DCMH 参考实现（仅供查阅）
├── data/               # 数据加载和预处理
├── training/           # 训练流程
│   ├── trainer.py               # 训练器
│   ├── loss.py                  # 通用损失函数
│   ├── dcmh_loss.py             # DCMH 特定损失函数
│   └── scheduler.py             # 学习率调度器
├── evaluation/         # 评估指标和基准测试
├── utils/              # 工具函数
├── examples/           # 示例脚本
│   └── dcmh_aspe_demo.py        # DCMH + ASPE 集成演示
├── tests/              # 单元测试
│   └── test_aspe_dcmh.py        # DCMH 集成测试
├── checkpoints/        # 模型检查点
└── results/            # 评估结果
```

## 安装

```bash
# 克隆仓库
git clone <repository-url>
cd DeepHash-ASPE

# 安装依赖
pip install -r requirements.txt

# 对于 MS-COCO 数据集
pip install pycocotools
```

## DCMH 模型架构

DCMH (Deep Cross-Modal Hashing) 是一个双流深度哈希模型，将图像和文本映射到共同的哈希空间。

### 模型组件

#### 1. DCMHImageModule（图像编码模块）

使用 AlexNet 风格的 CNN 架构：

```python
from core.hashing.dcmh_image import DCMHImageModule

# 创建图像模块
image_module = DCMHImageModule(bit=64, pretrain_model=None)

# 前向传播
image = torch.randn(4, 3, 224, 224)  # [batch, 3, H, W]
hash_codes = image_module(image)     # [batch, bit]
```

**架构细节**：
- 5 个卷积层 + 2 个全卷积层
- LocalResponseNorm 归一化
- MaxPool 下采样
- 输出：连续哈希码（可通过 sign() 转换为二进制）

#### 2. DCMHTextModule（文本编码模块）

使用两个卷积层处理文本标签：

```python
from core.hashing.dcmh_text import DCMHTextModule

# 创建文本模块
text_module = DCMHTextModule(y_dim=1000, bit=64)

# 准备输入（multi-hot 标签编码）
labels = torch.zeros(4, 1, 1000, 1)  # [batch, 1, y_dim, 1]
labels[0, 0, [1, 5, 10], 0] = 1.0    # 激活标签

# 前向传播
hash_codes = text_module(labels)     # [batch, bit]
```

**架构细节**：
- Conv2d(1, 8192, kernel_size=(y_dim, 1))
- ReLU 激活
- Conv2d(8192, bit, kernel_size=1)
- 输出：连续哈希码

#### 3. DCMHModel（双流模型）

整合图像和文本模块的统一接口：

```python
from core.hashing.dcmh_model import DCMHModel

# 创建模型
model = DCMHModel(bit=64, y_dim=1000, image_pretrain=None)

# 编码图像
img_hash = model.encode_image(image)

# 编码文本
txt_hash = model.encode_text(text_labels)

# 跨模态检索
similarity = model.compute_similarity(img_hash, txt_hash)
```

### 损失函数

```python
from training.dcmh_loss import DCMHCombinedLoss

# 创建损失函数
criterion = DCMHCombinedLoss(
    margin=1.0,
    quantization_weight=0.1,
    use_info_nce=False
)

# 计算损失
total_loss, loss_dict = criterion(image_hash, text_hash)
```

**损失组成**：
- **跨模态对比损失**：对齐图像和文本哈希码
- **量化损失**：鼓励输出接近 {-1, +1}

### 模型配置

```python
from config.model_config import DCMH_CONFIG

print(DCMH_CONFIG)
# {
#     'bit': 64,                # 哈希码位数
#     'y_dim': 1000,            # 文本标签维度
#     'image_pretrain': None,   # 预训练路径
#     'quantization_tau': 1.0,  # 量化温度
#     'with_quantization': True # 使用量化训练
# }
```

### 使用示例

```python
import torch
from core.hashing.dcmh_model import DCMHModel
from core.hashing.dcmh_text import DCMHTextModule

# 创建模型
model = DCMHModel(bit=64, y_dim=80)  # MS-COCO 有 80 个类别

# 准备输入
batch_size = 4
image = torch.randn(batch_size, 3, 224, 224)

# multi-hot 标签（80 个类别）
labels = torch.zeros(batch_size, 1, 80, 1)
for i in range(batch_size):
    num_active = torch.randint(1, 5, (1,)).item()
    active = torch.randperm(80)[:num_active]
    labels[i, 0, active, 0] = 1.0

# 生成哈希码
img_hash = model.encode_image(image)
txt_hash = model.encode_text(labels)

# 二进制哈希码
binary_img = torch.sign(img_hash)
binary_txt = torch.sign(txt_hash)

print(f"图像哈希：{img_hash.shape}")
print(f"文本哈希：{txt_hash.shape}")
print(f"二进制图像哈希唯一值：{torch.unique(binary_img)}")
```

---

## 快速开始

### 1. 在 MS-COCO 上训练

```python
from examples import train_on_coco

# 首先更新 config/coco_config.py 中的路径
train_on_coco.main()
```

### 2. 构建加密数据库

```python
from examples import build_encrypted_db

build_encrypted_db.main()
```

### 3. 执行检索

```python
from examples import text_to_image, image_to_text

# 文本 → 图像检索
text_to_image.main()

# 图像 → 文本检索
image_to_text.main()
```

## 使用示例

### 基础检索流程

```python
import torch
from core.hashing.dual_stream import DualStreamHashModel
from core.retrieval.pipeline import RetrievalPipeline

# 加载模型
model = DualStreamHashModel(vocab_size=10000, feature_dim=4096)

# 创建检索流程
pipeline = RetrievalPipeline(model, aspe_scheme='scheme1')

# 准备数据库
pipeline.prepare_database(images=image_list, texts=text_list)

# 执行检索
results = pipeline.text_to_image(query_text, k=10)
for result in results:
    print(f"得分: {result['score']}, 元数据: {result['metadata']}")
```

### 直接使用 ASPE

```python
import numpy as np
from core.aspe.scheme1 import ASPEScheme1

# 初始化 ASPE
aspe = ASPEScheme1(d=128)

# 加密数据库
db_features = np.random.randn(100, 128)
encrypted_db = aspe.encrypt_database(db_features)

# 加密查询
query = np.random.randn(128)
trapdoor = aspe.encrypt_query(query)

# 在加密空间计算相似度
score = aspe.ciphertext_inner_product(encrypted_db[0], trapdoor)
```

### DCMH + ASPE 集成

系统支持将 DCMH（Deep Cross-Modal Hashing）生成的哈希码进行 ASPE 加密，实现隐私保护的跨模态检索。

#### 运行演示

```bash
python examples/dcmh_aspe_demo.py
```

#### 使用 ASPEForDCMH 包装器

```python
import numpy as np
from core.aspe.dcmh_wrapper import ASPEForDCMH

# 初始化 ASPE for DCMH（指定哈希位数）
aspe = ASPEForDCMH(bit_dim=64, seed=42)

# 假设已有 DCMH 生成的 {-1, +1} 哈希码
retrieval_codes = np.sign(np.random.randn(1000, 64))  # 检索库
query_codes = np.sign(np.random.randn(100, 64))       # 查询

# 加密检索库
encrypted_db = aspe.GenEnc(retrieval_codes)

# 加密查询（生成陷阱门）
encrypted_query = aspe.GenTrap(query_codes)

# 计算密文汉明距离
distances = aspe.ciphertext_hamming_distance(encrypted_query, encrypted_db)

# 计算密文 mAP
query_labels = np.random.randint(0, 2, (100, 10))
db_labels = np.random.randint(0, 2, (1000, 10))
map_score = aspe.calc_ciphertext_map(
    encrypted_query, encrypted_db, query_labels, db_labels
)

print(f"mAP: {map_score:.4f}")
```

#### 数学原理

ASPE 方案 1 加密后，密文内积与原始内积的关系：
```
cipher_ip = p·q - 0.5×bit  （对于{-1, +1}哈希码）
```

汉明距离与内积的关系：
```
hamm = 0.5×(bit - p·q) = 0.25×bit - 0.5×cipher_ip
```

因此，**密文汉明距离排序 = 原始汉明距离排序**，mAP 值保持不变（允许并列值导致的微小误差 < 1e-3）。

## 组件

### 深度哈希模型

#### DCMH 模型（推荐）
- `DCMHImageModule`：AlexNet 风格的 CNN 图像编码器
- `DCMHTextModule`：基于卷积的文本标签编码器
- `DCMHModel`：统一的双流哈希模型
- `DCMHWithQuantization`：带量化训练的 DCMH 模型

#### 旧版模型（保留兼容）
- `ImageHashModel`：基于 CNN 的图像编码器（ResNet50/VGG16）
- `TextHashModel`：基于 NLP 的文本编码器（Transformer/LSTM）
- `DualStreamHashModel`：用于公共嵌入空间的统一模型

### ASPE 加密
- `ASPEScheme1`：基础 2 级安全（抵抗已知样本攻击）
- `ASPEScheme2`：增强 3 级安全（抵抗已知明文攻击）
- `ASPEForDCMH`：DCMH 专用包装器（支持 {-1, +1} 哈希码）

### 检索
- `SecureRetrievalEngine`：隐私保护相似度搜索
- `RetrievalPipeline`：端到端检索流程

### 损失函数
- `DCMHCombinedLoss`：DCMH 组合损失（跨模态 + 量化）
- `DCMHInfoNCELoss`：InfoNCE 对比损失
- `DCMHMarginRankingLoss`：边界排序损失

## 评估指标

系统支持标准检索评估：
- **mAP**：平均精度均值
- **Precision@K**：前 K 个结果的精度
- **Recall@K**：前 K 个结果的召回率
- **NDCG@K**：归一化折扣累积增益

## 配置

主要配置文件：
- `config/model_config.py`：模型架构设置
- `config/aspe_config.py`：ASPE 加密设置
- `config/train_config.py`：训练参数
- `config/coco_config.py`：MS-COCO 数据集路径

## 数据集

系统设计用于 MS-COCO 2014：
- 从 [COCO 数据集](https://cocodataset.org/)下载
- 在 `config/coco_config.py` 中更新路径

## 安全属性

### 方案 1（2 级）
- 抵抗已知样本攻击
- 距离不可恢复
- 陷阱门不可链接

### 方案 2（3 级）
- 抵抗已知明文攻击
- 随机拆分防止矩阵恢复
- 人工维度增强安全性

## 性能

在 MS-COCO 上的典型性能：
- **加密**：每个向量约 1-5ms（取决于维度）
- **查询**：前 100 个检索约 10-50ms
- **存储**：由于维度扩展，开销约 2 倍

## 引用

如果您使用此代码，请引用：

```bibtex
@software{deephash_aspe,
  title={DeepHash-ASPE: 隐私保护跨模态检索系统},
  author={您的名字},
  year={2025},
  url={https://github.com/yourusername/DeepHash-ASPE}
}
```

## 许可证

MIT 许可证 - 详见 LICENSE 文件

---

## DCMH + ASPE 演示系统

### 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Next.js 14 前端                          │
├─────────────────────────────────────────────────────────────┤
│  页面：/ (首页) | /demo | /dataset | /encrypt | /metrics    │
└─────────────────────────────────────────────────────────────┘
                              ↓ HTTP/REST API
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI 后端                              │
├─────────────────────────────────────────────────────────────┤
│  API: /api/images | /api/texts | /api/hash | /api/encrypt  │
│       /api/search | /api/metrics                             │
│  服务：DCMHService | ASPEService | COCOService              │
└─────────────────────────────────────────────────────────────┘
```

### 安装

```bash
# 1. 安装后端依赖
cd backend
pip install -r requirements.txt

# 2. 安装前端依赖
cd ../frontend
npm install
```

### 启动服务

```bash
# 终端 1：启动 FastAPI 后端
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 终端 2：启动 Next.js 前端
cd frontend
npm run dev
```

### 访问应用

- 前端：http://localhost:3000
- API 文档：http://localhost:8000/docs
- Redoc: http://localhost:8000/redoc

### 页面说明

| 页面 | 路由 | 功能 |
|------|------|------|
| 首页 | `/` | 系统介绍和快速入口 |
| 演示 | `/demo` | 交互式跨模态检索 |
| 数据集 | `/dataset` | MS-COCO 数据集浏览 |
| 加密可视化 | `/encrypt` | ASPE 加密过程演示 |
| 指标 | `/metrics` | mAP 对比和性能分析 |

### API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/images/upload` | POST | 上传图像 |
| `/api/images/feature` | POST | 提取图像特征 |
| `/api/texts/process` | POST | 处理文本 |
| `/api/hash/image` | POST | 生成图像哈希码 |
| `/api/hash/text` | POST | 生成文本哈希码 |
| `/api/encrypt/database` | POST | 加密数据库 |
| `/api/encrypt/trapdoor` | POST | 生成陷阱门 |
| `/api/search` | POST | 执行检索 |
| `/api/metrics` | GET | 获取评估指标 |

### 技术栈

**前端**
- Next.js 14 (App Router)
- React 18
- TypeScript
- Tailwind CSS
- Recharts (图表)
- Framer Motion (动画)
- React Dropzone (文件上传)

**后端**
- FastAPI
- Pydantic
- PyTorch
- NumPy
- DCMH 参考实现
- ASPE 加密方案

---

## 致谢

- MS-COCO 数据集创建者
- PyTorch 团队
- ASPE 加密方案研究者
