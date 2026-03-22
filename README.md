# DeepHash-ASPE：隐私保护跨模态检索系统

一个结合了深度学习哈希与非对称标量积保持加密（ASPE）的隐私保护图文检索系统。

## 📢 更新 (2026-03-22)

### CNN+ASPE 隐私检索系统完整实现

已完成 **CNN 图像检索 + SkNN 隐私保护检索** 的完整实现和模块化整合：

**新增模块**：

1. **core/cirtorch/** - 完整复制自 cnnimageretrieval-pytorch-master2
   - `examples/` - 训练和测试脚本 (`train.py`, `test.py`, `test_e2e.py`)
   - `scripts/` - 工具脚本 (`build_db.py` - 加密数据库构建)
   - `api/` - FastAPI 路由端点 (`endpoints.py`)
   - `services/` - 服务层 (`sknn_service.py` - SkNN 隐私检索服务)
   - `layers/`, `networks/`, `datasets/`, `utils/` - 核心库代码

2. **core/aspe/cnn_wrapper.py** - CNN 特征的 ASPE 加密包装器
   - `ASPEForCNN` 类：支持 CNN 特征向量的 SkNN 加密
   - 内积保持性验证
   - 密钥生成/加载/保存

3. **backend/app/routers/cir_retrieval.py** - 更新的 API 路由
   - 明文检索端点（传统 CNN 相似度）
   - SkNN 隐私检索端点（加密保护）
   - 数据库构建和管理端点

**SkNN 加密原理**：
```
建库端：S=0 时复制，S=1 时随机拆分
查询端：S=0 时拆分 (r=0)，S=1 时复制
密文内积 = 明文内积（保持性）
```

**快速使用**：
```python
# 方式 1: 使用 SknnService
from cirtorch.services.sknn_service import SknnService

service = SknnService(feature_dim=2048, model_path='model.pth')

# 构建加密数据库
db_features, db_images = service.build_database(
    image_dir='./images',
    save_dir='./data/retrieval_db'
)

# 隐私保护检索
results = service.search(query_image_path='query.jpg', top_k=10)

# 方式 2: 使用 ASPEForCNN
from core.aspe.cnn_wrapper import ASPEForCNN

aspe = ASPEForCNN(feature_dim=2048)
aspe.generate_keys()  # 生成密钥

# 加密数据库特征
encrypted_db = aspe.encrypt_database(features)  # [N, 2d]

# 加密查询
enc_query = aspe.encrypt_query(query_feature)  # [2d]

# 密文检索
scores, indices = aspe.search(encrypted_db, query_feature, top_k=10)
```

**API 端点**：
- `GET /api/cir/status` - 服务状态
- `POST /api/cir/search` - 明文/密文检索
- `POST /api/cir/sknn/keys/generate` - 生成 SkNN 密钥
- `POST /api/cir/sknn/database/build` - 构建加密数据库
- `POST /api/cir/sknn/search` - SkNN 隐私检索
- `POST /api/cir/sknn/search/upload` - 上传图像隐私检索

详见：
- [core/cirtorch](core/cirtorch/) - CNN 检索核心库
- [core/aspe/cnn_wrapper.py](core/aspe/cnn_wrapper.py) - ASPE 加密包装器
- [backend/app/routers/cir_retrieval.py](backend/app/routers/cir_retrieval.py) - API 路由

---

### Flickr25K 完整系统测试和实验

已添加 **Flickr25K 数据集完整测试和实验脚本**：

- 🚀 **一键运行**：执行完整训练 + 评估流程
- 📊 **实时可视化**：mAP 对比、Precision@K、训练曲线
- 📝 **实验报告**：自动生成 Markdown 格式报告
- 🔐 **ASPE 测试**：内积保持性、排序一致性、mAP 保持性验证

**快速启动**：
```bash
# Windows
run_flickr25k_experiment.bat

# Linux/Mac
./run_flickr25k_experiment.sh

# 或手动运行
python experiments/run_flickr25k.py --data data/flickr25k/FLICKR-25K.mat --bit 64 --epochs 500
```

**输出结果**（`results/flickr-25k/` 目录）：
- `flickr25k_experiment_report.md` - 完整实验报告
- `flickr25k_experiment_results.json` - 详细数据
- `*.png` - 可视化图表
- `dcmh_best.pth` - 最佳模型检查点

详见 [Flickr25K 实验](#flickr25k-实验)

---

## 📢 更新 (2026-03-21)

### 前端 UI/UX 美化 (2026-03-21)

已完成 **DeepHash-ASPE 演示系统前端 UI/UX 全面美化**：

**设计系统**
- 🎨 **配色方案**：Indigo 主题色 (#6366F1) + Emerald 辅助色
- 📐 **字体配对**：Fira Sans (正文) + Fira Code (代码/数据)
- ✨ **设计风格**：扁平化设计，无阴影/渐变，简洁线条

**组件美化**
- **Navbar**：玻璃态效果、SVG 图标、激活状态指示器
- **ImageUpload**：改进拖拽反馈、缩放预览效果
- **ResultsGrid**：排名徽章、悬停效果、渐变进度条
- **MetricsChart**：自定义工具提示、优化图表配色

**页面美化**
- **首页**：渐变色 Logo、特性卡片图标、改进 CTA 按钮
- **演示页**：分段式类型切换器、徽章样式标签
- **数据集页**：统计卡片图标化、信息层级优化
- **加密可视化**：SVG 图标替代 emoji、步骤动画优化
- **指标页**：关键指标卡片、统一图表风格

**技术改进**
- ✅ 无 emoji 图标（全部使用 SVG）
- ✅ 所有可点击元素添加 cursor-pointer
- ✅ 平滑过渡动画 (150-300ms)
- ✅ 响应式设计 (375px - 1440px)
- ✅ 无障碍访问支持

详见 [frontend/src/app/globals.css](frontend/src/app/globals.css) 和各组件文件。

---

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

## Flickr25K 实验

### 概述

[Flickr25K](https://forms.illinois.edu/sec/2298948) 是一个包含 25,015 张图像的跨模态检索数据集，每张图像配有文本描述和 24 个类别标签。

**数据集统计**：
| 划分 | 样本数 |
|------|--------|
| 查询集 | 2,000 |
| 训练集 | 10,000 |
| 数据库 | 18,015 |
| 类别数 | 24 |

### 快速启动

**Windows**：
```bash
run_flickr25k_experiment.bat
```

**Linux/Mac**：
```bash
chmod +x run_flickr25k_experiment.sh
./run_flickr25k_experiment.sh
```

**手动运行**：
```bash
python experiments/run_flickr25k.py \
    --data data/flickr25k/FLICKR-25K.mat \
    --bit 64 \
    --epochs 500 \
    --batch-size 128 \
    --lr 1e-4 \
    --result-dir results/flickr-25k
```

### 配置选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `data/flickr25k/FLICKR-25K.mat` | 数据文件路径 |
| `--bit` | 64 | 哈希码维度 |
| `--epochs` | 500 | 最大训练轮数 |
| `--batch-size` | 128 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--result-dir` | `results/flickr-25k` | 结果输出目录 |
| `--no-gpu` | - | 禁用 GPU |

### 输出结果

实验完成后，`results/flickr-25k/` 目录包含：

**报告和数据**：
- `flickr25k_experiment_report.md` - Markdown 格式实验报告
- `flickr25k_experiment_results.json` - JSON 格式详细数据

**可视化图表**：
- `flickr25k_map_comparison.png` - mAP 对比图
- `flickr25k_precision_k.png` - Precision@K 对比图
- `flickr25k_training_curve.png` - 训练损失和 mAP 曲线
- `flickr25k_aspe_test.png` - ASPE 加密测试结果

**模型检查点**：
- `dcmh_best.pth` - 最佳模型（验证 mAP 最高）
- `dcmh_final.pth` - 最终模型
- `dcmh_checkpoint_*.pth` - 每 50 轮保存的检查点

### 实验流程

实验脚本自动执行以下步骤：

1. **数据加载** - 从 FLICKR-25K.mat 加载图像、文本和标签
2. **DCMH 训练** - 训练双流哈希模型（图像 + 文本）
3. **检索评估** - 计算 mAP、Precision@K、Recall@K
4. **ASPE 测试** - 验证内积保持性、排序一致性、mAP 保持性
5. **可视化** - 生成图表和 Markdown 报告

### 预期性能

在 Flickr25K 数据集上的典型性能（bit=64）：
- **图像→文本 mAP**: ~0.55-0.65
- **文本→图像 mAP**: ~0.50-0.60
- **平均 mAP**: ~0.55

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
├── config/                 # 配置文件
│   ├── __init__.py
│   ├── model_config.py     # 模型配置（DCMH、CNN）
│   ├── aspe_config.py      # ASPE 加密配置
│   ├── train_config.py     # 训练参数配置
│   ├── dataset_config.py   # 数据集配置（Flickr25K, NUS-WIDE 等）
│   ├── eval_config.py      # 评估配置
│   └── dcmh_config.py      # DCMH 专用配置
│
├── core/                   # 核心库
│   ├── aspe/               # ASPE 加密方案
│   │   ├── __init__.py
│   │   ├── scheme1.py      # 基础 2 级安全方案
│   │   ├── scheme2.py      # 增强 3 级安全方案
│   │   ├── keygen.py       # 密钥生成
│   │   ├── dcmh_wrapper.py # DCMH 集成包装器
│   │   └── cnn_wrapper.py  # CNN 集成包装器
│   │
│   ├── hashing/            # 深度哈希模型
│   │   ├── __init__.py
│   │   ├── dcmh_image.py   # DCMH 图像编码器 (AlexNet CNN)
│   │   ├── dcmh_text.py    # DCMH 文本编码器 (卷积标签处理)
│   │   ├── dcmh_model.py   # DCMH 双流模型
│   │   └── dcmh_basic.py   # DCMH 基础模块
│   │
│   ├── cirtorch/           # CNN 图像检索 (复制自 reference)
│   │   ├── layers/         # pooling, normalization, loss
│   │   ├── networks/       # ImageRetrievalNet
│   │   ├── datasets/       # 数据加载器
│   │   ├── utils/          # 评估、whitening、下载
│   │   ├── api/            # FastAPI 端点
│   │   └── services/       # 服务层 (SkNN 检索)
│   │
│   └── retrieval/          # 检索工具
│       └── dcmh_metrics.py # DCMH 评估指标
│
├── reference/              # 参考实现（只读）
│   ├── DCMH/                       # DCMH 原始代码
│   └── cnnimageretrieval-pytorch-master2/  # cirtorch 原始代码
│
├── training/               # 训练流程
│   ├── __init__.py
│   ├── trainer.py          # 通用训练器
│   ├── dcmh_loss.py        # DCMH 专用损失函数
│   ├── scheduler.py        # 学习率调度器
│   └── train_flickr25k.py  # Flickr25K 训练脚本
│
├── evaluation/             # 评估工具
│   ├── __init__.py
│   ├── metrics.py          # 通用评估指标
│   ├── security.py         # 安全属性评估
│   ├── benchmark.py        # 性能基准测试
│   ├── evaluate_flickr25k.py   # Flickr25K 评估
│   └── quick_eval_dcmh_aspe.py # DCMH+ASPE 快速评估
│
├── experiments/            # 完整实验脚本
│   ├── __init__.py
│   └── run_flickr25k.py    # Flickr25K 完整实验
│
├── examples/               # 示例脚本
│   ├── __init__.py
│   ├── dcmh_aspe_demo.py   # DCMH+ASPE 集成演示
│   ├── build_encrypted_db.py # 构建加密数据库
│   ├── text_to_image.py    # 文本→图像检索示例
│   └── image_to_text.py    # 图像→文本检索示例
│
├── tests/                  # 单元测试
│   ├── __init__.py
│   ├── test_aspe_dcmh.py       # ASPE+DCMH 集成测试
│   ├── test_cnn_aspe.py        # CNN+ASPE 集成测试
│   └── test_dataset_downloader.py # 数据集下载测试
│
├── backend/                # FastAPI 后端
│   ├── app/
│   │   ├── main.py
│   │   ├── routers/        # API 路由
│   │   └── services/       # 业务服务
│   └── start_backend.py
│
├── frontend/               # Next.js 14 前端
│   ├── src/
│   ├── public/
│   └── package.json
│
├── data/                   # 数据集
│   ├── FLICKR-25K.mat      # Flickr25K 数据集
│   └── imagenet-vgg-f.mat  # CNN 预训练模型
│
├── checkpoints/            # 模型检查点
├── results/                # 评估结果
├── utils/                  # 通用工具
│   ├── __init__.py
│   ├── crypto.py           # 加密工具
│   └── matrix.py           # 矩阵运算工具
│
├── run_flickr25k_experiment.bat/sh  # 快捷实验脚本
├── download_datasets.bat/sh         # 快捷下载脚本
├── start_all.bat/sh                 # 启动所有服务
└── README.md
```

> **注**：项目已于 2026-03-22 完成重构，删除了以下冗余文件：
> - 旧版哈希模型：`image_hash.py`, `text_hash.py`, `dual_stream.py`
> - 旧版检索模块：`cir_service.py`, `feature_extractor.py`, `pipeline.py`, `secure_retrieval.py`
> - 重复脚本：`quick_download.py`, `quick_flickr25k.py`, `run_flickr25k.py`
> - 冗余测试：`test_dcmh_full.py`, `test_dcmh_aspe_comprehensive.py`
> - 重复损失函数：`training/loss.py`

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

### 1. 训练 DCMH 模型（Flickr25K）

```bash
# 使用脚本（Windows）
run_flickr25k_experiment.bat

# 使用脚本（Linux/Mac）
./run_flickr25k_experiment.sh

# 或直接使用 Python
python training/train_flickr25k.py --data data/FLICKR-25K.mat --bit 64 --epochs 100
```

### 2. 运行完整实验

```bash
# 完整训练 + 评估 + 可视化
python experiments/run_flickr25k.py --bit 64 --epochs 500
```

### 3. 快速评估 DCMH+ASPE

```bash
python evaluation/quick_eval_dcmh_aspe.py --mode quick --bit 64
```

### 4. 查看演示示例

```bash
# DCMH+ASPE 集成演示
python examples/dcmh_aspe_demo.py

# CNN+ASPE 测试
python tests/test_cnn_aspe.py
```

## 使用示例


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

### ASPE 加密
- `ASPEScheme1`：基础 2 级安全（抵抗已知样本攻击）
- `ASPEScheme2`：增强 3 级安全（抵抗已知明文攻击）
- `ASPEForDCMH`：DCMH 专用包装器（支持 {-1, +1} 哈希码）
- `ASPEForCNN`：CNN 专用包装器（支持 SkNN 加密）

### 检索
- `SecureRetrievalEngine`：隐私保护相似度搜索

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

### 系统测试数据集

DCMH + ASPE 系统使用**类别标签 multi-hot 编码**作为文本输入，而非自然语言文本。
这意味着只需要**图像 + 类别标签**，不需要复杂的文本标注。

#### 推荐数据集

| 优先级 | 数据集 | 标签数 | 图像数 | 推荐理由 |
|--------|--------|--------|--------|----------|
| 1 | **Flickr-25K** | 24 | 25K | 参考实现原生支持，DCMH 论文基准 |
| 2 | **IAPR TC-12** | 255 | 20K | 多标签基准，图像质量高 |
| 3 | **NUS-WIDE** | 81 | 270K | 多标签检索标准基准 |
| 4 | **MS-COCO** | 80 | 118K | 代码已内置支持 |

#### 下载方法

**方法 1：使用下载脚本（推荐）**

```bash
# Windows
download_datasets.bat

# Linux/Mac
./download_datasets.sh
```

**方法 2：手动下载**

##### Flickr-25K（推荐）
- 百度网盘：https://pan.baidu.com/s/1X5BTyux524aUyqHpFGPPlA
- 提取码：`eico`
- 目标路径：`data/flickr25k/FLICKR-25K.mat`

##### IAPR TC-12
- 官网：http://www.imageclef.org/photodata
- 需要注册并申请下载
- 目标路径：`data/iapr_tc12/IAPR-TC12.mat`

##### NUS-WIDE
- 官网：http://lms.comp.nus.edu.sg/research/nuswide.shtml
- 下载 ImageList.txt 和 Groundtruth.zip
- 目标路径：`data/nuswide/NUS-WIDE.mat`

#### 数据集配置

在 `config/dataset_config.py` 中配置数据集路径：

```python
from config.dataset_config import get_dataset_config, check_dataset_exists

# 获取 Flickr-25K 配置
config = get_dataset_config('flickr25k')
print(config['data_path'])  # ./data/flickr25k/FLICKR-25K.mat
print(config['n_classes'])  # 24
print(config['n_images'])   # 25015

# 检查数据集是否存在
exists, path, msg = check_dataset_exists('flickr25k')
if exists:
    print(f"数据集已就绪：{path}")
else:
    print(f"需要下载数据集：{path}")
```

### MS-COCO 2014（备选）

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
