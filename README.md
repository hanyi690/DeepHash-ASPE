# DeepHash-ASPE：隐私保护跨模态检索系统

## 功能特性

- **跨模态检索**：文本 ↔ 图像搜索（DCMH 深度跨模态哈希）
- **图像检索**：相似图像搜索（CNN 特征 + SkNN 隐私保护）
- **隐私保护**：使用 ASPE 加密实现加密数据库检索
- **全检索模式**：文本→图像、图像→文本、图像→图像
- **结果可视化**：统一使用 ResultsGrid 组件显示检索结果（含图像缩略图）
- **高质量图像显示**：DCMH 检索结果直接从原始 JPG 文件加载，避免预处理图像恢复带来的质量损失
- **Web 演示系统**：Next.js 14 前端 + FastAPI 后端

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Next.js 14 前端                           │
│  页面：/ (首页) | /search (统一检索)                         │
└─────────────────────────────────────────────────────────────┘
                              ↓ HTTP/REST API
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI 后端                              │
│  API: /api/images | /api/texts | /api/hash | /api/encrypt  │
│       /api/search (统一检索) | /api/tags | /api/datasets   │
│       /cir-images | /flickr-images (静态文件服务)           │
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
│   │   ├── keygen.py       # 密钥生成
│   │   ├── dcmh_wrapper.py # DCMH 哈希加密包装器 (SkNN 风格)
│   │   └── cnn_wrapper.py  # CNN 特征加密包装器 (SkNN)
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
│   ├── infer_tag_mapping.py   # 推断 YAll 列映射
│   ├── build_all_cache.py     # 构建所有缓存
│   ├── verify_tag_mapping.py  # 验证标签映射
│   └── clean_flickr25k_dataset.py # 清理 Flickr25k 数据集
│
├── backend/                # FastAPI 后端
│   └── app/
│       ├── main.py
│       ├── routers/        # API 路由
│       └── services/       # 业务服务
│
├── frontend/               # Next.js 14 前端
│   └── src/
│       ├── app/
│       │   ├── page.tsx        # 首页
│       │   └── search/         # 统一检索页面
│       │       └── page.tsx    # 标签搜图/图搜标签/图搜图
│       └── components/         # 共享组件
│           ├── ImageUpload.tsx # 图像上传组件
│           ├── ResultsGrid.tsx # 结果网格（显示图片）
│           ├── DatasetSelector.tsx # 数据集选择器
│           └── LabelSelector.tsx   # 标签选择器
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

### 数据一致性说明

Flickr25K 数据集的训练数据 `FLICKR-25K.mat` 和原始标签文件 `common_tags.txt` 存在**列顺序不一致**的问题：

- `FLICKR-25K.mat` 中的 `YAll` 矩阵列顺序与 `common_tags.txt` 行顺序不对应
- 原因：数据来源不同，DCMH 旧版数据的标签顺序已被打乱

系统通过**频率匹配**推断列映射：

```bash
# 推断 YAll 列索引 -> tag 名称的映射
python scripts/infer_tag_mapping.py

# 输出: data/dcmh/flickr25k/tag_mapping.npy
```

**映射原理**：

1. 统计 YAll 每列的 1 的数量
2. 读取 common_tags.txt 中每个 tag 的频率
3. 使用匈牙利算法匹配，最小化差异
4. 验证：高频 tag（explore、sky、nikon 等）映射差异 < 35

**映射文件**：

- `tag_mapping.npy`：YAll 列索引 -> tag 名称的列表
- `tag_mapping.txt`：文本版本，便于查看

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

### 构建 CIR 缓存（可选）

CIR 服务支持懒加载模式，首次请求时自动加载数据。如需预先构建缓存：

```bash
# 构建 CIR 缓存（自动使用配置文件中的正确路径）
python scripts/build_all_cache.py --type cir --dataset roxford5k

# 构建所有缓存
python scripts/build_all_cache.py --type all
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

# ASPE 加密 (SkNN 风格，密文内积 = 明文内积)
aspe = ASPEForDCMH(bit_dim=64)
# 密钥自动生成: M1, M2, S 矩阵

encrypted_db = aspe.GenEnc(database_hashes)      # 加密数据库
trapdoor = aspe.GenTrap(query_hash)              # 生成陷阱门

# 密文检索（密文内积 = 明文内积，mAP 完全一致）
hamming_dist = aspe.ciphertext_hamming_distance(trapdoor, encrypted_db)
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
| 标签数 | 1,386  |

**数据格式**：

系统从原始 JPG 图像和标签文件加载数据：

```
data/dcmh/flickr25k/
├── mirflickr/               # 原始图像目录
│   └── im1.jpg ~ im25000.jpg
├── doc/
│   └── common_tags.txt      # 标签名列表（1386个）
├── meta/                    # 元数据目录
├── img_model.pth            # 图像编码器（部署后）
├── txt_model.pth            # 文本编码器（部署后）
└── tag_mapping.npy          # 标签索引映射
```

**数据集文件**：

- `FLICKR-25K.mat`：位于 `data/` 根目录，包含预处理后的图像数据、YAll、LAll
- `mirflickr/`：原始 JPG 图像，用于高质量显示检索结果

**下载方式**：

- 百度网盘：https://pan.baidu.com/s/1X5BTyux524aUyqHpFGPPlA
- 提取码：`eico`
- 下载后解压到 `data/dcmh/flickr25k/` 目录

### Flickr25K Cleaned（清理后的数据集）

原始 Flickr25k 数据集中很多图片的标签不在 `common_tags.txt` 中。系统提供了清理脚本，生成过滤后的数据集：

| 指标           | 数值    |
| -------------- | ------- |
| 原始图片数     | 25,000  |
| 有效图片数     | 20,359  |
| 无效图片数     | 4,641   |
| 过滤前总标签数 | 223,537 |
| 过滤后总标签数 | 94,282  |

**目录结构**：

```
data/dcmh/flickr25k_cleaned/
├── images/              # 图片文件（复制，保留原名 im1.jpg 等）
├── tags/                # 清理后的标签文件（只含 common_tags）
├── index_mapping.json   # 有效图片的原始 idx 列表
└── metadata.json        # 数据集统计信息
```

**运行清理脚本**：

```bash
python scripts/clean_flickr25k_dataset.py
```

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

## 缓存结构

系统使用预计算缓存加速检索，缓存存储在 `backend/cache/` 目录：

```
backend/cache/
├── dcmh/                          # DCMH 缓存
│   └── flickr25k/
│       ├── database_image.npz     # 图像哈希码 [18015, 64]（4.4MB）
│       ├── database_text.npz      # 文本哈希码 [18015, 64]（4.4MB）
│       ├── encrypted.npz          # 加密数据库 [18015, 128]（18MB）
│       ├── lall.npz               # LAll 类别标签 [18015, 24]（3.5MB）
│       ├── query.npz              # 查询数据 [2000, ...]（11MB）
│       └── tags.npz               # YAll 标签向量 [18015, 1386]（200MB）
│
└── cir/                           # CIR 缓存
    ├── roxford5k/
    │   ├── features.npz           # 明文特征 [N, 2048]（42MB）
    │   └── encrypted.npz          # 加密特征 [N, 4096]（83MB）
    └── rparis6k/
        ├── features.npz
        └── encrypted.npz
```

**缓存说明**：

- `database_image.npz`：检索库图像哈希码，用于标签搜图
- `database_text.npz`：检索库文本哈希码，用于图搜标签
- `lall.npz`：24维类别标签，用于 mAP 计算
- `tags.npz`：1386维标签向量，用于检索结果显示标签名称
- `encrypted.npz`：ASPE 加密后的特征/哈希码（维度翻倍）

**缓存构建**：

```bash
# 构建所有缓存
python scripts/build_all_cache.py --type all

# 仅构建 DCMH 缓存
python scripts/build_all_cache.py --type dcmh --dataset flickr25k

# 强制重建
python scripts/build_all_cache.py --type dcmh --force
```

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

## 数据存储结构

```
data/
├── FLICKR-25K.mat                          # Flickr25K 数据集（3.3GB）
├── imagenet-vgg-f.mat                      # DCMH 预训练权重
│
├── dcmh/                                   # DCMH 数据和模型
│   └── flickr25k/
│       ├── mirflickr/                      # 原始图像目录
│       │   └── im1.jpg ~ im25000.jpg
│       ├── doc/
│       │   └── common_tags.txt             # 标签名列表（1386个）
│       ├── meta/                           # 元数据
│       │   ├── tags/, tags_raw/
│       │   ├── exif/, exif_raw/
│       │   └── license/
│       ├── img_model.pth                   # 图像编码器（228MB）
│       ├── txt_model.pth                   # 文本编码器（47MB）
│       ├── tag_mapping.npy                 # 标签索引映射
│       ├── tag_mapping.txt                 # 标签映射文本版
│       ├── training_result.json            # 训练配置
│       └── deploy_record.json              # 部署记录
│
├── networks/                               # 预训练模型
│   ├── imagenet-caffe-resnet50-features-ac468af.pth
│   ├── imagenet-caffe-resnet101-features-10a101d.pth
│   └── gl18-tl-resnet101-gem-w-a4d43db.pth # 推荐（187MB）
│
├── whiten/                                 # 白化权重
│   └── retrieval-SfM-120k-resnet101-gem-whiten-22ab0c1.pth
│
└── test/                                   # CIR 评估数据集
    ├── roxford5k/
    │   ├── jpg/                            # 图像
    │   └── gnd_roxford5k.pkl               # Ground Truth
    └── rparis6k/
        ├── jpg/
        └── gnd_rparis6k.pkl
```

## 检索模式

系统支持三种检索模式：

| 模式                     | 说明                       | 技术方案                 |
| ------------------------ | -------------------------- | ------------------------ |
| **标签搜图** (T2I) | 选择标签索引，检索相关图像 | DCMH 深度跨模态哈希      |
| **图搜标签** (I2T) | 上传图像，预测相关标签     | DCMH 跨模态检索          |
| **图搜图** (CIR)   | 上传图像，检索相似图像     | CNN 特征 + SkNN 隐私保护 |

### 检索流程

#### 标签搜图 (T2I)

1. 前端发送 YAll 列索引（0-1385）
2. 后端构建 1386 维 multi-hot 向量（对应位置设为 1）
3. 使用 DCMH 文本编码器生成哈希码
4. 在图像哈希码数据库中计算汉明距离
5. 返回 Top-K 相似图像

#### 图搜标签 (I2T)

1. 前端上传图像（base64 编码）
2. 后端预处理图像（224x224，RGB）
3. 使用 DCMH 图像编码器生成哈希码
4. 在文本哈希码数据库中计算汉明距离
5. 返回 Top-K 相似图像的标签

**注意**：两种模式使用不同的数据库：

- 标签搜图：使用图像哈希码数据库
- 图搜标签：使用文本哈希码数据库

## ASPE 加密原理

系统使用 ASPE 算法（SkNN 风格）：

### ASPE 算法 (SkNN 风格)

- **特性**: 密文内积 = 明文内积
- **密钥**: 双矩阵 M1, M2 + 拆分向量 S
- **密文维度**: 2×bit (无扩展)
- **拆分策略**: SkNN 风格互补拆分
  - 建库端: S=0 复制, S=1 随机拆分
  - 查询端: S=0 固定拆分(r=0), S=1 复制
- **适用场景**: 高精度跨模态检索

**加密原理**:

```
建库端：S=0 复制，S=1 随机拆分 → v1, v2
查询端：S=0 拆分 (r=0)，S=1 复制 → w1, w2
密文内积 = v1·w1 + v2·w2 = v·w（明文内积，完全保持）
```

**安全属性**：

- 距离不可恢复
- 陷阱门不可链接
- 密文 mAP = 明文 mAP（完全一致）

## 数据计算方式说明

### 汉明距离计算

**明文汉明距离**：

```python
# 哈希码格式：{-1, +1}^{N × bit}
# 汉明距离与内积的线性关系：
hamming_dist = 0.5 × (bit - inner_product)

# 示例：bit=64
# 完全相同：内积=64，距离=0
# 完全相反：内积=-64，距离=64
# 随机码：内积≈0，距离≈32
```

**密文汉明距离（ASPE 加密）**：

```python
# 密文内积 = 明文内积（ASPE 核心特性）
encrypted_inner = q1 @ r1.T + q2 @ r2.T  # 密文内积

# 密文汉明距离 = 明文汉明距离
hamming_dist = 0.5 × (bit - encrypted_inner)
```

### mAP 计算

**GPU 加速版本**：

```python
def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    """
    计算跨模态检索的 mAP@K。

    参数：
        qB: {-1,+1}^{m×q} 查询哈希码
        rB: {-1,+1}^{n×q} 检索库哈希码
        query_L: {0,1}^{m×l} 查询标签（LAll 类别标签）
        retrieval_L: {0,1}^{n×l} 检索库标签（LAll 类别标签）

    返回：
        mAP 值
    """
    # 1. 计算汉明距离矩阵 [m, n]
    hamm = 0.5 × (bit - qB @ rB.T)

    # 2. 计算相关性矩阵 [m, n]
    # 如果查询标签与检索库标签有交集，则为相关
    gnd = (query_L @ retrieval_L.T > 0)

    # 3. 对每个查询计算 AP
    for each query:
        # 按距离排序（升序）
        sorted_indices = argsort(hamm)

        # 计算平均精度
        AP = mean(precision at each relevant position)

    # 4. 返回所有查询的 mAP
    return mean(all APs)
```

**关键点**：

- 使用 **LAll（24维类别标签）** 计算 mAP，而非 YAll（1386维文本标签）
- 相关性定义：查询与检索库至少有一个共同类别

### 命中率计算

前端返回的 `HitStats` 包含两种命中率：

```typescript
interface HitStats {
  // 标签命中（YAll）
  tag_hits: number;          // 命中 YAll 标签的结果数
  tag_hit_rate: number;      // tag_hits / total_results

  // 类别命中（LAll）- 与评估 mAP 对应
  category_hits: number;     // 命中 LAll 类别的结果数
  category_hit_rate: number; // category_hits / total_results

  // 查询信息
  query_tags: number[];      // 查询的 YAll 索引
  query_tag_names: string[]; // 查询标签名称
}
```

**计算方式**：

```python
# YAll 标签命中：检索结果包含查询标签
tag_hit = any(query_tag in result_tags)

# LAll 类别命中：检索结果与查询有共同类别
# 这与 mAP 计算的相关性定义一致
category_hit = any((result_LAll > 0) & (query_LAll > 0))
```

### CIR 检索计算

#### 特征提取

CIR 使用 CNN 模型（如 ResNet101-GeM）提取图像特征：

```python
# 特征维度：2048（ResNet101-GeM）
# 图像预处理：Resize → Tensor → Normalize
features = model.extract_vectors(images)  # [N, 2048]
```

#### 相似度计算

CIR 使用**内积**作为相似度度量（非汉明距离）：

```python
# 明文相似度矩阵 [Q, N]
similarities = query_features @ database_features.T

# 排序：降序（相似度越高越相似）
ranks = np.argsort(-similarities, axis=1)
```

#### 密文相似度（ASPE 加密）

```python
# 加密后维度：2 × feature_dim = 4096
encrypted_db = aspe.encrypt_database(db_features)    # [N, 4096]
encrypted_query = aspe.encrypt_query(query_feature)  # [4096]

# 密文内积 = 明文内积
cipher_similarities = encrypted_db @ encrypted_query.T
```

#### CIR mAP 计算（Revisited 协议）

ROxford5k/RParis6k 使用三种难度的 mAP 评估：

```python
# Easy: 只考虑 easy 标注作为正例
gnd['ok'] = gnd[i]['easy']
gnd['junk'] = gnd[i]['junk'] + gnd[i]['hard']

# Medium: easy + hard 作为正例
gnd['ok'] = gnd[i]['easy'] + gnd[i]['hard']
gnd['junk'] = gnd[i]['junk']

# Hard: 只考虑 hard 标注作为正例
gnd['ok'] = gnd[i]['hard']
gnd['junk'] = gnd[i]['junk'] + gnd[i]['easy']
```

**关键点**：

- junk 标注不计入 AP 计算（既不是正例也不是负例）
- 密文 mAP 与明文 mAP 完全一致

## API 返回数据说明

### `/api/search` 返回结构

```typescript
interface UnifiedSearchResponse {
  success: boolean;              // 请求是否成功
  query_type: string;            // 'tag_to_image' | 'image_to_tag'
  tag_indices?: number[];        // 查询的 YAll 标签索引
  query_tag_names?: string[];    // 查询标签名称
  results: SearchResult[];       // 标签搜图结果
  tag_results: ImageToTagResult[]; // 图搜标签结果
  total_results: number;         // 结果总数
  search_time_ms: number;        // 搜索耗时（毫秒）
  hit_stats?: HitStats;          // 命中率统计
  encryption_info?: EncryptionInfo; // 加密信息
  query_hash_code?: number[];    // 查询哈希码（用于调试）
}
```

### `SearchResult` 字段说明

```typescript
interface SearchResult {
  rank: number;              // 排名（1-based）
  image_id: number;          // 图像 ID
  score: number;             // 相似度分数 = 1 / (1 + distance)
  distance: number;          // 汉明距离（0 ~ bit）
  tags: number[];            // 该图像的 YAll 标签索引
  tag_names: string[];       // 标签名称
  hit_tags: number[];        // 命中的查询标签索引
  hit_tag_names: string[];   // 命中的标签名称
  thumbnail_url?: string;    // 缩略图 URL
  hash_code?: number[];      // 该图像的哈希码
  category_hit: boolean;     // LAll 类别是否命中
  tag_hit: boolean;          // YAll 标签是否命中
}
```

### `HitStats` 计算方式

```typescript
interface HitStats {
  total_results: number;     // 返回的结果数量

  // YAll 标签命中
  tag_hits: number;          // tag_hit=true 的结果数
  tag_hit_rate: number;      // tag_hits / total_results

  // LAll 类别命中（与 mAP 相关性一致）
  category_hits: number;     // category_hit=true 的结果数
  category_hit_rate: number; // category_hits / total_results

  // 查询信息
  query_tags: number[];      // 查询的 YAll 索引
  query_tag_names: string[]; // 查询标签名称
}
```

### `EncryptionInfo` 内容

```typescript
interface EncryptionInfo {
  method: string;            // 加密方法：'ASPE (SkNN)'
  query_encrypted: boolean;  // 查询是否加密
  database_encrypted: boolean; // 数据库是否加密
  security_level: number;    // 安全级别
  bit_dim: number;           // 哈希码位数
}
```

## API 端点

### 统一检索端点（推荐）

| 端点                     | 方法 | 描述                  |
| ------------------------ | ---- | --------------------- |
| `/api/search`          | POST | 统一检索（JSON 请求） |
| `/api/search/upload`   | POST | 统一检索（文件上传）  |
| `/api/search/status`   | GET  | 获取服务状态          |
| `/api/search/datasets` | GET  | 获取支持的数据集列表  |

### 检索模式说明

| 模式               | 数据集              | 描述             |
| ------------------ | ------------------- | ---------------- |
| `tag_to_image`   | flickr25k, nuswide  | 标签搜图（DCMH） |
| `image_to_tag`   | flickr25k, nuswide  | 图搜标签（DCMH） |
| `image_to_image` | roxford5k, rparis6k | 图搜图（CIR）    |

### 统一请求格式

```json
{
  "mode": "tag_to_image",
  "encryption": "encrypted",
  "dataset": "flickr25k",
  "top_k": 10,
  "tag_indices": [0, 1, 2]
}
```

### 其他端点

| 端点                 | 方法 | 描述         |
| -------------------- | ---- | ------------ |
| `/api/images/{id}` | GET  | 获取图像     |
| `/api/tags/names`  | GET  | 获取标签列表 |
| `/api/encrypt/status` | GET | 获取加密服务状态 |

## 性能

### DCMH 在 Flickr25K 上的性能（bit=64）

| 指标      | 值                        |
| --------- | ------------------------- |
| mAP(i→t) | **0.7409**          |
| mAP(t→i) | **0.7716**          |
| 平均 mAP  | **0.7563**          |
| 训练时间  | ~180min (RTX 4060 laptop) |

### ASPE 验证结果 (SkNN 风格)

| 指标      | 明文 mAP | 密文 mAP | 误差          |
| --------- | -------- | -------- | ------------- |
| mAP(i→t) | 0.740915 | 0.740915 | **0.0** |
| mAP(t→i) | 0.771607 | 0.771607 | **0.0** |

**密文维度**: 128 (原 64 → 直接拼接为 2×64)
**特性**: 密文内积 = 明文内积，mAP 完全一致

### 排序一致性验证

| 指标           | 值      |
| -------------- | ------- |
| Top-10 交集率  | 100%    |
| Top-50 交集率  | 100%    |
| Top-100 交集率 | 100%    |
| 最大距离误差   | < 1e-12 |

### 哈希码质量

| 指标         | 值     |
| ------------ | ------ |
| 平衡性       | 0.9098 |
| 唯一性       | 0.767  |
| 平均汉明距离 | 0.4949 |

## 配置

主要配置文件：

- `config/dcmh_config.py`：DCMH 模型和训练参数
- `config/aspe_config.py`：ASPE 加密参数
- `config/dataset_config.py`：数据集路径配置

## 更新日志

### 2026-03-27: 清理冗余代码和统一架构完善

**目标**：完成统一架构重构后的清理工作，移除不再使用的旧实现。

**删除的文件**：

| 文件 | 原因 |
|------|------|
| `backend/app/routers/search.py` | 被 unified_search.py 替代 |
| `backend/app/routers/cir_retrieval.py` | 被 unified_search.py 替代 |
| `backend/app/routers/metrics.py` | 前端未调用 |
| `backend/app/services/aspe_service.py` | 被 dcmh_encryption_service.py 替代 |
| `frontend/src/components/MetricsChart.tsx` | 未被使用 |

**更新的文件**：

| 文件 | 变更 |
|------|------|
| `backend/app/main.py` | 移除旧路由注册，仅保留统一端点 |
| `backend/app/routers/encrypt.py` | 改用 DCMHEncryptionService |
| `backend/app/routers/metrics.py` | 已删除 |
| `backend/app/schemas/search.py` | 移除未使用的 schema |
| `backend/app/schemas/__init__.py` | 更新导出 |
| `backend/app/services/__init__.py` | 更新服务导出 |

**当前路由结构**：

```
app.include_router(images.router)
app.include_router(texts.router)
app.include_router(hash_codes.router)
app.include_router(encrypt.router)
app.include_router(unified_search.router)  # 统一检索端点
app.include_router(tags.router)
app.include_router(datasets.router)
```

**保留的底层服务**：

| 服务 | 作用 | 被谁使用 |
|------|------|----------|
| `dcmh_service.py` | DCMH 模型加载、哈希码生成 | `dcmh_search_service.py` |
| `cir_service.py` | CNN 模型加载、特征提取 | `cir_search_service.py` |
| `hash_cache_service.py` | 哈希码缓存管理 | `dcmh_search_service.py` |

### 2026-03-27: 统一 DCMH 和 CIR 系统架构

**目标**：统一 DCMH 跨模态检索和 CIR 图像检索两个子系统的架构，提高代码一致性和可维护性。

**主要变更**：

1. **统一类型定义**（`backend/app/schemas/unified.py`）：

   - `SearchMode` 枚举：`tag_to_image`、`image_to_tag`、`image_to_image`
   - `EncryptionMode` 枚举：`plaintext`、`encrypted`
   - 统一结果类型：`BaseSearchResult`、`TagToImageResult`、`ImageToTagResult`、`ImageToImageResult`
   - 统一请求/响应类型：`UnifiedSearchRequest`、`UnifiedSearchResponse`
2. **加密服务抽象**：

   - `BaseEncryptionService`：加密服务抽象基类
   - `DCMHEncryptionService`：DCMH 哈希码加密服务
   - `CIREncryptionService`：CIR CNN 特征加密服务
3. **检索服务抽象**：

   - `BaseSearchService`：检索服务抽象基类
   - `DCMHSearchService`：DCMH 跨模态检索服务
   - `CIRSearchService`：CIR 图像检索服务
   - `SearchServiceFactory`：服务工厂
4. **统一端点**（`backend/app/routers/unified_search.py`）：

   - `POST /api/search`：JSON 请求
   - `POST /api/search/upload`：文件上传
   - `GET /api/search/status`：服务状态
   - `GET /api/search/datasets`：数据集列表
5. **前端统一**（`frontend/src/lib/api.ts`）：

   - 统一类型定义与后端一致
   - `unifiedSearch()`：JSON 请求
   - `unifiedSearchUpload()`：文件上传

**新架构**：

```
┌─────────────────────────────────────────────────────────────┐
│                      前端 (Next.js)                          │
│  unifiedSearch() / unifiedSearchUpload()                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 统一端点 /api/search                         │
│  mode: tag_to_image | image_to_tag | image_to_image        │
│  encryption: plaintext | encrypted                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  SearchServiceFactory                        │
└─────────────────────────────────────────────────────────────┘
                    ↓                       ↓
┌──────────────────────────┐    ┌──────────────────────────┐
│   DCMHSearchService      │    │   CIRSearchService       │
│   (flickr25k, nuswide)   │    │   (roxford5k, rparis6k) │
├──────────────────────────┤    ├──────────────────────────┤
│ - DCMHEncryptionService  │    │ - CIREncryptionService   │
│ - DCMHService (特征)     │    │ - CIRService (特征)      │
└──────────────────────────┘    └──────────────────────────┘
```

**文件变更**：

| 文件                                                | 操作   |
| --------------------------------------------------- | ------ |
| `backend/app/schemas/unified.py`                  | 新建   |
| `backend/app/services/base_encryption_service.py` | 新建   |
| `backend/app/services/base_search_service.py`     | 新建   |
| `backend/app/services/dcmh_encryption_service.py` | 新建   |
| `backend/app/services/cir_encryption_service.py`  | 新建   |
| `backend/app/services/dcmh_search_service.py`     | 新建   |
| `backend/app/services/cir_search_service.py`      | 新建   |
| `backend/app/routers/unified_search.py`           | 新建   |
| `frontend/src/lib/api.ts`                         | 重构   |
| `frontend/src/app/search/page.tsx`                | 重构   |
| `backend/app/main.py`                             | 修改   |
| `backend/app/routers/search.py`                   | 已删除 |
| `backend/app/routers/cir_retrieval.py`            | 已删除 |
| `backend/app/routers/metrics.py`                  | 已删除 |
| `backend/app/services/aspe_service.py`            | 已删除 |
| `frontend/src/components/MetricsChart.tsx`        | 已删除 |

### 2026-03-26: 加密检索数据库修复

**问题**：图像→标签检索时，加密模式使用了错误的数据库（加密的图像哈希码而非加密的文本哈希码），导致明文和密文检索结果不一致。

**修复**：

- `hash_cache_service.py`：新增 `encrypted_text_database` 属性和相关存取方法，分别存储图像和文本加密数据库
- `aspe_service.py`：新增 `encrypt_text_database()` 方法专门加密文本哈希码，添加 `encrypted_text_database` 属性
- `search.py`：根据检索类型选择正确的加密数据库
  - 标签→图像检索：使用加密的**图像哈希码**数据库
  - 图像→标签检索：使用加密的**文本哈希码**数据库

**加密逻辑澄清**：

- 加密**只需要哈希码本身**，不需要标签数据
- LAll（类别标签）仅用于 mAP 评估，与加密无关
- YAll（文本标签）用于生成文本哈希码和显示标签名称

**缓存结构更新**：

```
backend/cache/dcmh/flickr25k/
├── encrypted.npz          # 加密的图像哈希码（用于标签→图像检索）
└── encrypted_text.npz     # 加密的文本哈希码（用于图像→标签检索）
```

### 2026-03-26: 排序一致性修复

**问题**：明文和密文 mAP 计算使用了不同的排序策略，导致排序一致性验证失败。

**修复**：

- `evaluation/metrics.py`：`calc_map_k` 函数统一使用 `np.round(decimals=10) + np.lexsort`
- `backend/app/routers/search.py`：Top-K 排序统一使用 `np.round + np.lexsort`
- `backend/app/services/aspe_service.py`：`_compute_map_from_distances` 统一排序逻辑
- `core/aspe/dcmh_wrapper.py`：新增 `verify_sorting_consistency()` 方法验证排序一致性

**验证结果**：

| 指标           | 值   |
| -------------- | ---- |
| Top-10 交集率  | 100% |
| Top-50 交集率  | 100% |
| Top-100 交集率 | 100% |

### 2026-03-26: 图像预处理修复

**问题**：检索时的图像预处理减了两次均值（预处理函数 + 模型 forward），而训练时只减一次（仅 forward）。

**修复**：

- `backend/app/services/dcmh_service.py`：移除 `preprocess_image_for_inference` 中的预减均值逻辑
- 预处理现在只做：RGB转换、resize、float32转换、HWC→CHW
- 均值减法由模型 forward 统一处理

**预处理流程对比**：

| 步骤         | 训练时      | 检索时（修复后） |
| ------------ | ----------- | ---------------- |
| 加载图像     | h5 直接加载 | PIL resize       |
| 预处理       | 无          | 无               |
| forward      | 减 mean     | 减 mean          |
| 总减均值次数 | 1次         | 1次              |

### 2026-03-26: README 重写

**更新内容**：

- 更新性能数据：mAP 从 ~0.54 更新为实际值 0.7409 / 0.7716
- 新增"数据计算方式说明"章节：汉明距离、mAP、命中率计算公式
- 新增 CIR 检索计算说明：特征提取、相似度计算、Revisited 协议 mAP
- 新增"API 返回数据说明"章节：详细说明返回结构和字段含义
- 更新数据存储结构：与实际目录一致，包含完整文件路径
- 更新缓存结构：修正 DCMH 缓存（lall.npz, tags.npz）和 CIR 缓存文件列表
- 更新项目结构：移除已删除的文件（scheme1.py, scheme2.py, benchmark.py, security.py）
- 增强 ASPE 验证结果：新增排序一致性验证数据

### 2026-03-26: 图像预处理与颜色修复

**问题 1：图像颜色异常**

- 原因：`FLICKR-25K.mat` 中图像是 BGR 格式，已减去 VGG-F 均值
- 修复：`_load_mat_image` 使用正确的 VGG-F 均值恢复和 BGR→RGB 转换

**问题 2：图搜文检索效果差**

- 原因：用户上传图片的预处理与训练数据不一致
- 训练时：原始图像 → 减去 VGG-F 均值 → 存入 .mat → 模型 forward 再减均值
- 用户上传：原始图像 → 直接传入模型 → 只减一次均值
- 修复：`preprocess_image_for_inference` 添加 VGG-F 均值减法步骤

**VGG-F 均值（BGR 格式）**：

- B = 123.66, G = 116.77, R = 103.93

**修改文件**：

- `backend/app/services/dcmh_service.py`：修复预处理函数，添加均值减法
- `backend/app/routers/images.py`：修复图像显示，使用正确均值和 BGR→RGB

### 2026-03-26: 代码清理与精简

**目标**：移除冗余代码，提高代码可读性和可维护性。

**清理内容**：

1. **P0 未使用代码删除**：

   - `dcmh_service.py`：删除 `PRETRAIN_MODEL_PATH`、`DEFAULT_Y_DIM`、`preprocess_tag_vector()`、`generate_database_codes()`、`get_all_dcmh_services()`
   - `dataset_service.py`：删除 `get_all_dataset_services()`
   - `cir_service.py`：删除 `get_feature_dim()`
   - `search.py`：删除 `_generate_demo_query_code()`
2. **P1 错误代码删除**：

   - `cir_retrieval.py`：删除调用不存在方法的端点 `build_index()`、`save_index()`、`load_index()`
   - 删除相关未使用的请求模型：`BuildIndexRequest`、`LoadIndexRequest`、`SaveIndexRequest`、`SknnBuildDatabaseRequest`
3. **P2 兼容旧格式代码删除**：

   - `hash_cache_service.py`：删除 `save_cache()`、`load_cache()`、`save_encrypted_cache()`、`load_encrypted_cache()`、`build_database_cache()`、`get_cache_info_legacy()`
   - `hash_cache_service.py`：简化 `load_dcmh_tags()`，移除旧格式检测逻辑
   - `schemas/search.py`：删除 `HitStats` 中的兼容字段 `hits`、`hit_rate`、`query_tag_count`

**API 更新**：

- `datasets.py` 和 `encrypt.py` 改用 `build_full_database_cache()` 替代已删除的 `build_database_cache()`

### 2026-03-26: DCMH 检索数据混淆修复

**问题**：`aspe_service.database_labels` 存储的是 YAll（1386维标签向量）而非 LAll（24维类别标签），导致 mAP 计算使用错误的数据。

**修复**：

- `search.py`：`ensure_cache_initialized` 函数现在正确加载 LAll 赋值给 `aspe_service.database_labels`
- `hash_cache_service.py`：`build_encrypted_cache` 方法现在使用 LAll 进行加密
- `hash_cache_service.py`：新增 `load_dcmh_yall` 方法，返回 YAll 数据（`load_full_database` 作为兼容别名保留）

**数据命名规范**：

- `YAll (tags)`：标签向量（1386维）- 用于生成文本哈希码、检索结果显示标签名称
- `LAll (labels)`：类别标签向量（24维）- 用于计算 mAP

**数据流**：

```
hash_cache.load_dcmh_yall() → YAll
hash_cache.load_dcmh_lall() → LAll

hash_cache.database_tags     = YAll (用于检索结果显示)
aspe_service.database_labels = LAll (用于 mAP 计算)
```

## 许可证

MIT 许可证
