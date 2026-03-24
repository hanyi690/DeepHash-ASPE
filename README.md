# DeepHash-ASPE：隐私保护跨模态检索系统

一个结合了深度学习哈希与非对称标量积保持加密（ASPE）的隐私保护图文检索系统。

## 功能特性

- **跨模态检索**：文本 ↔ 图像搜索（DCMH 深度跨模态哈希）
- **图像检索**：相似图像搜索（CNN 特征 + SkNN 隐私保护）
- **隐私保护**：使用 ASPE 加密实现加密数据库检索
- **全检索模式**：文本→图像、图像→文本、图像→图像
- **Web 演示系统**：Next.js 14 前端 + FastAPI 后端

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Next.js 14 前端                           │
│  页面：/ (首页) | /demo | /search | /cir | /dataset | /encrypt | /metrics │
└─────────────────────────────────────────────────────────────┘
                              ↓ HTTP/REST API
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI 后端                              │
│  API: /api/images | /api/texts | /api/hash | /api/encrypt  │
│       /api/search | /api/cir | /api/metrics                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     核心模块                                  │
│  - core/hashing: DCMH 深度哈希模型                           │
│  - core/cirtorch: CNN 图像检索                               │
│  - core/aspe: ASPE 加密方案                                  │
│  - evaluation: 统一评估器                                    │
└─────────────────────────────────────────────────────────────┘
```

## 项目结构

```
DeepHash-ASPE/
├── config/                 # 配置文件
│   ├── dcmh_config.py      # DCMH 训练配置
│   ├── aspe_config.py      # ASPE 加密配置
│   ├── dataset_config.py   # 数据集配置
│   └── coco_config.py      # COCO 数据集路径
│
├── core/                   # 核心库
│   ├── aspe/               # ASPE 加密方案
│   │   ├── scheme1.py      # 基础 2 级安全方案
│   │   ├── scheme2.py      # 增强 3 级安全方案
│   │   ├── keygen.py       # 密钥生成
│   │   ├── dcmh_wrapper.py # DCMH 哈希加密包装器
│   │   └── cnn_wrapper.py  # CNN 特征加密包装器
│   │
│   ├── hashing/            # 深度哈希模型
│   │   ├── dcmh_image.py   # DCMH 图像编码器
│   │   ├── dcmh_text.py    # DCMH 文本编码器
│   │   ├── dcmh_model.py   # DCMH 双流模型
│   │   └── dcmh_data_loader.py  # 数据加载
│   │
│   └── cirtorch/           # CNN 图像检索
│       ├── layers/         # 池化层、归一化层
│       ├── networks/       # ImageRetrievalNet
│       ├── datasets/       # 数据加载器
│       └── utils/          # 评估、白化工具
│
├── training/               # 训练流程
│   ├── train_dcmh.py       # DCMH 训练脚本
│   └── dcmh_dataset.py     # 按需加载数据集
│
├── evaluation/             # 评估工具
│   ├── metrics.py          # 核心指标（mAP, P@K, R@K）
│   ├── evaluator.py        # DCMH 统一评估器
│   ├── cir_evaluator.py    # CIR 统一评估器
│   └── visualization.py    # 可视化函数
│
├── scripts/                # 工具脚本
│   ├── evaluate.py         # 统一评估入口
│   ├── deploy_model.py     # 模型部署脚本
│   ├── download_cir_model.py  # CIR 模型下载
│   ├── download_cir_dataset.py # CIR 数据集下载
│   ├── build_cir_db.py     # 构建检索数据库
│   └── rebuild_hash_cache.py  # 重建哈希缓存
│
├── backend/                # FastAPI 后端
│   └── app/
│       ├── main.py
│       ├── routers/        # API 路由
│       └── services/       # 业务服务
│
├── frontend/               # Next.js 14 前端
│   └── src/app/
│       ├── page.tsx        # 首页
│       ├── demo/           # 演示页面
│       ├── search/         # 统一检索页面
│       ├── cir/            # CNN 检索页面
│       ├── dataset/        # 数据集浏览
│       ├── encrypt/        # 加密可视化
│       └── metrics/        # 指标展示
│
├── tests/                  # 单元测试
├── results/                # 训练结果和评估报告
└── data/                   # 数据集目录
```

## 安装

### 1. 创建虚拟环境

```bash
# 克隆仓库
git clone <repository-url>
cd DeepHash-ASPE

# 创建虚拟环境
python -m venv .venv

# Windows 激活
.venv\Scripts\activate

# Linux/Mac 激活
source .venv/bin/activate
```

### 2. 安装 PyTorch GPU

```bash
# CUDA 12.4（推荐）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 验证 GPU
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available())"
```

### 3. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 4. 安装前端依赖

```bash
cd frontend
npm install
```

## 快速开始

### 训练 DCMH 模型

```bash
# 使用 GPU 训练
python training/train_dcmh.py train --bit 64 --epochs 500

# 低内存模式（按需加载，内存 < 100MB）
python training/train_dcmh.py train --bit 64 --epochs 500 --batch_size 64

# 恢复训练
python training/train_dcmh.py train --resume_from results/flickr-25k/最新时间戳
```

### 模型部署

训练完成后，将模型部署到 `data/dcmh/` 目录供后端服务调用：

```bash
# 部署最新训练的模型
python scripts/deploy_model.py --dataset flickr25k

# 部署指定训练结果
python scripts/deploy_model.py --dataset flickr25k --result_dir results/flickr-25k/20260324_090727

# 强制覆盖已部署的模型
python scripts/deploy_model.py --dataset flickr25k --force
```

部署后的目录结构：

```
data/dcmh/flickr25k/
├── img_model.pth          # 图像编码器
├── txt_model.pth          # 文本编码器
├── training_result.json   # 训练配置
└── deploy_record.json     # 部署记录
```

### 评估模型

```bash
# DCMH 评估（包含 ASPE 验证）
python scripts/evaluate.py --result_dir results/flickr-25k/最新时间戳

# CIR 评估（默认启用 ASPE 验证）
python -m evaluation.cir_evaluator \
    --model_path data/networks/gl18-tl-resnet101-gem-w-a4d43db.pth \
    --data_dir data/test/roxford5k \
    --dataset roxford5k

# CIR 评估（禁用 ASPE 验证）
python -m evaluation.cir_evaluator \
    --model_path data/networks/gl18-tl-resnet101-gem-w-a4d43db.pth \
    --data_dir data/test/roxford5k \
    --dataset roxford5k \
    --no-aspe
```

### 启动演示系统

```bash
# 终端 1：启动后端
cd backend
python -m uvicorn app.main:app --reload --port 8000

# 终端 2：启动前端
cd frontend
npm run dev

# 访问：http://localhost:3000
```

## 使用示例

### Python API

#### DCMH + ASPE 跨模态检索

```python
from core.hashing.dcmh_model import DCMHModel
from core.aspe.dcmh_wrapper import ASPEForDCMH

# 创建 DCMH 模型
model = DCMHModel(bit=64, y_dim=24)  # Flickr25K 有 24 个类别

# 生成哈希码
img_hash = model.encode_image(image)
txt_hash = model.encode_text(text_labels)

# ASPE 加密
aspe = ASPEForDCMH(bit_dim=64)
aspe.generate_keys()

encrypted_db = aspe.GenEnc(database_hashes)      # 加密数据库
trapdoor = aspe.GenTrap(query_hash)              # 生成陷阱门

# 密文检索（mAP 与明文相等）
scores, indices = aspe.search(encrypted_db, query_hash, top_k=10)
```

#### CNN + SkNN 图像检索

```python
from core.cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from core.aspe.cnn_wrapper import ASPEForCNN

# 加载预训练模型
state = torch.load('model.pth')
model = init_network(state['meta'])
model.load_state_dict(state['state_dict'])

# 提取特征
features = extract_vectors(model, images, image_size, transform)

# SkNN 加密
aspe = ASPEForCNN(feature_dim=2048)
aspe.generate_keys()

encrypted_db = aspe.encrypt_database(db_features)
trapdoor = aspe.encrypt_query(query_feature)

# 密文检索
scores, indices = aspe.search(encrypted_db, query_feature, top_k=10)
```

### 统一评估器

```python
from evaluation import DCMHEvaluator, CIREvaluator

# DCMH 评估
dcmh_eval = DCMHEvaluator(result_dir='results/flickr-25k/20260324_090727')
results = dcmh_eval.evaluate(run_aspe=True)

# CIR 评估
cir_eval = CIREvaluator(
    model_path='data/networks/gl18-tl-resnet101-gem-w-a4d43db.pth',
    data_dir='data/test/roxford5k',
    dataset='roxford5k'
)
results = cir_eval.evaluate()  # 默认启用 ASPE 验证
# 或禁用: results = cir_eval.evaluate(run_aspe=False)
```

## 数据集

### Flickr25K（推荐用于 DCMH）

| 划分   | 样本数 |
| ------ | ------ |
| 查询集 | 2,000  |
| 训练集 | 10,000 |
| 数据库 | 18,015 |
| 类别数 | 24     |

**下载方式**：

- 百度网盘：https://pan.baidu.com/s/1X5BTyux524aUyqHpFGPPlA
- 提取码：`eico`
- 目标路径：`data/flickr25k/FLICKR-25K.mat`

### ROxford5k / RParis6k（用于 CIR 评估）

Revisited 版图像检索基准数据集，用于评估 CNN 特征检索性能。

| 数据集    | 图像数 | 查询数 | 来源             |
| --------- | ------ | ------ | ---------------- |
| ROxford5k | 5,062  | 70     | Oxford Buildings |
| RParis6k  | 6,392  | 70     | Paris Buildings  |

**下载方式**：

```bash
# 推荐：设置代理后使用官方源（最稳定）
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890
python scripts/download_cir_dataset.py --dataset roxford5k --mirror official

# 或使用官方源（国内网络可能不稳定）
python scripts/download_cir_dataset.py --dataset roxford5k --mirror official

# 自动模式：优先尝试 HuggingFace，失败后回退官方源
python scripts/download_cir_dataset.py --dataset roxford5k

# 下载 RParis6k
python scripts/download_cir_dataset.py --dataset rparis6k

# 下载所有
python scripts/download_cir_dataset.py --all
```

**下载加速建议**：

1. 使用代理：设置 `HTTP_PROXY` 和 `HTTPS_PROXY` 环境变量
2. 手动下载：从浏览器下载后放到对应目录
   - Oxford5k 图像: https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images-v1.tgz
   - Paris6k 图像: https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/
   - Ground Truth: http://cmp.felk.cvut.cz/cnnimageretrieval/data/test/

## 预训练模型

### CIR 检索模型

| 模型                  |    ROxf(M)    |    RPar(M)    | 说明           |
| --------------------- | :------------: | :------------: | -------------- |
| GL18-ResNet101-GeM-W  | **67.3** | **80.6** | 推荐，最高性能 |
| SfM120k-ResNet101-GeM |      65.4      |      76.7      | 备选           |

**下载方式**：

```bash
# 下载推荐模型（国内网络较慢，支持断点续传）
python scripts/download_cir_model.py --model gl18-resnet101-gem-w

# 如果下载中断，再次运行会从断点继续
python scripts/download_cir_model.py --model gl18-resnet101-gem-w

# 列出所有可用模型
python scripts/download_cir_model.py --list
```

**国内加速方案**：

1. 使用代理：设置 `HTTP_PROXY` 和 `HTTPS_PROXY` 环境变量
2. 手动下载：从浏览器下载后放到 `data/networks/` 目录
   - GL18 模型：http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth

### 模型存储结构

```
data/
├── dcmh/                                  # DCMH 部署模型
│   └── flickr25k/
│       ├── img_model.pth                  # 图像编码器
│       ├── txt_model.pth                  # 文本编码器
│       └── deploy_record.json             # 部署记录
│
├── networks/                              # 预训练模型
│   ├── imagenet-caffe-resnet50-features-ac468af.pth   # ResNet50 特征
│   ├── imagenet-caffe-resnet101-features-10a101d.pth  # ResNet101 特征
│   └── gl18-tl-resnet101-gem-w-a4d43db.pth            # 推荐，最高性能
│
├── whiten/                                # 白化权重
│   └── retrieval-SfM-120k-resnet101-gem-whiten-22ab0c1.pth  # ResNet101 白化
│
├── test/                                  # 评估数据集
│   ├── roxford5k/
│   │   ├── jpg/                           # 图像
│   │   └── gnd_roxford5k.pkl              # Ground Truth
│   └── rparis6k/
│       ├── jpg/
│       └── gnd_rparis6k.pkl
│
├── flickr25k/                             # DCMH 数据集
│   └── FLICKR-25K.mat
│
└── imagenet-vgg-f.mat                     # DCMH 预训练
```

## 检索模式

系统支持三种检索模式：

| 模式                     | 说明                       | 技术方案                 |
| ------------------------ | -------------------------- | ------------------------ |
| **标签搜图** (T2I) | 选择标签索引，检索相关图像 | DCMH 深度跨模态哈希      |
| **图搜标签** (I2T) | 上传图像，预测相关标签     | DCMH 跨模态检索          |
| **图搜图** (CIR)   | 上传图像，检索相似图像     | CNN 特征 + SkNN 隐私保护 |

## ASPE 加密原理

ASPE (Asymmetric Scalar-product-preserving Encryption) 是一种非对称标量积保持加密方案：

```
建库端：S=0 时复制，S=1 时随机拆分
查询端：S=0 时拆分 (r=0)，S=1 时复制
密文内积 = 明文内积（保持性）
```

**安全属性**：

- 距离不可恢复
- 陷阱门不可链接
- 密文 mAP = 明文 mAP

## API 端点

| 端点                            | 方法 | 描述          |
| ------------------------------- | ---- | ------------- |
| `/api/images/upload`          | POST | 上传图像      |
| `/api/search`                 | POST | 跨模态检索    |
| `/api/cir/search/upload`      | POST | CNN 图像检索  |
| `/api/cir/sknn/search/upload` | POST | SkNN 隐私检索 |
| `/api/metrics`                | GET  | 获取评估指标  |

## 性能

### DCMH 在 Flickr25K 上的典型性能（bit=64）

| 指标      | 值                |
| --------- | ----------------- |
| mAP(i→t) | ~0.54             |
| mAP(t→i) | ~0.54             |
| 训练时间  | ~30min (RTX 4060) |

### ASPE 验证结果

| 指标      | 明文 mAP | 密文 mAP | 误差    |
| --------- | -------- | -------- | ------- |
| mAP(i→t) | 0.5392   | 0.5392   | < 0.001 |
| mAP(t→i) | 0.5395   | 0.5394   | < 0.001 |

## 配置

主要配置文件：

- `config/dcmh_config.py`：DCMH 模型和训练参数
- `config/aspe_config.py`：ASPE 加密参数
- `config/dataset_config.py`：数据集路径配置

## 许可证

MIT 许可证
