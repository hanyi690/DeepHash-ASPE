# DeepHash-ASPE：隐私保护跨模态检索系统

一个结合了深度学习哈希与非对称标量积保持加密（ASPE）的隐私保护图文检索系统。

## 📢 更新 (2026-03-24)

### 验证过程 GPU 加速优化 ✅

已实现验证过程（哈希码生成 + mAP 计算）的 GPU 加速，验证速度提升 **10x+**。

**优化内容**：

| 瓶颈 | 原实现 | 优化后 | 提升 |
|------|--------|--------|------|
| 文本哈希码生成 | 逐样本处理 | 批处理 (batch_size=128) | **12x** |
| mAP 计算 | 逐查询计算汉明距离 | 一次性向量化计算 | **6x** |
| **总验证时间** | ~5min | ~30s | **10x** |

**修改文件**：
| 文件 | 修改 |
|------|------|
| `training/train_dcmh.py` | `generate_text_code_from_dataset` 批处理重构 |
| `core/retrieval/dcmh_metrics.py` | `calc_map_k` 向量化优化 |

**技术实现**：
```python
# 文本哈希码生成：逐样本 → 批处理
# 原：18015 次 GPU 调用
# 新：~140 次 GPU 调用（batch_size=128）
loader = DataLoader(dataset, batch_size=128)
for tags, labels, indices in loader:
    cur_g = txt_model(tags.cuda())  # 批量推理
    B[indices, :] = cur_g

# mAP 计算：向量化汉明距离
# 原：每个查询单独计算
# 新：一次性计算所有汉明距离矩阵 [num_query, num_retrieval]
hamm = calc_hammingDist(qB, rB)  # 已支持 GPU
gnd = (query_L.mm(retrieval_L.T) > 0).float()
```

**验证结果**：
- ✅ CPU/GPU mAP 计算结果一致（差异 < 1e-5）
- ✅ 批处理加速比 12.8x（2000 样本测试）

---

### DCMH 验证机制优化 ✅

已实现间隔验证机制，替代原有的 `valid` 布尔配置：

**改进内容**：

| 原配置 | 新配置 | 说明 |
|--------|--------|------|
| `valid=True` | `valid_interval=10` | 默认每 10 个 epoch 验证 |
| `valid=False` | `valid_interval=0` | 仅最终验证 |

**新增功能**：
- ✅ `mAP_history` 记录训练过程中 mAP 变化趋势
- ✅ 灵活配置验证频率，兼顾效率和监控能力

**修改文件**：
| 文件 | 修改 |
|------|------|
| `config/dcmh_config.py` | `valid` → `valid_interval` |
| `training/train_dcmh.py` | 间隔验证逻辑 + mAP 历史记录 |

**使用示例**：
```bash
# 每 5 个 epoch 验证
python training/train_dcmh.py train --valid_interval=5

# 每 epoch 验证（精确监控）
python training/train_dcmh.py train --valid_interval=1

# 仅最终验证
python training/train_dcmh.py train --valid_interval=0
```

**输出结果**：
`result.json` 中新增 `mAP_history` 列表，记录每次验证的 epoch 和 mAP 值。

---

## 📢 更新 (2026-03-23)

### 检查点加载卡住问题修复 ✅

已修复训练检查点加载时程序卡住无响应的问题：

**问题原因**：
- 检查点中的张量存储标记为 CUDA 设备
- `torch.load()` 在 Windows 上处理大文件和 CUDA 张量时可能出现死锁
- 大张量（F_buffer 144MB，B 64MB）加载时需要大量内存和设备同步

**修复内容**：

| 文件 | 修改 | 说明 |
|------|------|------|
| `training/train_dcmh.py` | `save_checkpoint` | 使用 `_use_new_zipfile_serialization=True` |
| `training/train_dcmh.py` | `load_checkpoint` | 添加 `weights_only=False` 和验证逻辑 |
| `training/train_dcmh.py` | `validate_checkpoint` | 新增检查点完整性验证函数 |
| `scripts/fix_checkpoint.py` | 新建 | 修复现有检查点文件的脚本 |

**修复现有检查点**：
```bash
# 修复现有检查点文件
python scripts/fix_checkpoint.py results/flickr-25k/20260323_183559/checkpoint.pth

# 保存到新文件
python scripts/fix_checkpoint.py results/flickr-25k/20260323_183559/checkpoint.pth --output results/flickr-25k/20260323_183559/checkpoint_fixed.pth
```

**恢复训练**：
```bash
python training/train_dcmh.py train --resume_from=results/flickr-25k/20260323_183559 --max_epoch=5
```

---

### ASPE 加密验证成功 ✅

已完成 **DCMH 模型训练和 ASPE 加密前后 mAP 对比评估**：

**训练结果**：
- 模型路径: `results/flickr-25k/20260323_103015/`
- 哈希码位数: 64 bits
- 训练轮数: 500 epochs

**ASPE 加密验证结果**：

| 指标 | 明文 mAP | 密文 mAP | 差异 |
|------|----------|----------|------|
| mAP(i→t) | 0.5392 | 0.5392 | 0.00003 |
| mAP(t→i) | 0.5395 | 0.5394 | 0.00011 |

**结论**: 明文和密文的 mAP 值几乎完全相等（差异 < 0.001），验证了 ASPE 加密方案正确保持了哈希码的排序关系，可以在密文状态下实现与明文相同的检索精度。

**生成文件**：
- `loss_curve.png` - 训练损失曲线
- `map_curve.png` - mAP 结果图
- `aspe_comparison.png` - 明文 vs 密文 mAP 对比图
- `aspe_evaluation.json` - 详细评估数据
- `report.md` - 完整评估报告

---

### 系统完整性检查修复

已完成**核心模块到后端再到前端的完整链路检查和修复**：

**修复内容**：

| 问题 | 文件 | 修复 |
|------|------|------|
| 后端服务导出缺失 | `backend/app/services/__init__.py` | 添加 CIRService 导出 |
| 明文检索端点缺失 | `backend/app/routers/cir_retrieval.py` | 新增 `/api/cir/search/upload` 端点 |
| EncryptionInfo 字段不完整 | `backend/app/routers/search.py` | 补充 method 和 security_level |
| CIR 模型加载需要显式路径 | `backend/app/services/cir_service.py` | 添加自动初始化 torchvision 预训练模型 |
| 字段名不一致 | `backend/app/services/cir_service.py` | image_path → image_name |

**新增功能**：
- ✅ CIR 服务支持自动加载 torchvision 预训练模型（无需手动下载）
- ✅ 明文检索模式端点完整支持
- ✅ 前后端字段名一致性保证

**验证通过**：
- CIRService 导入成功
- 端点 `/api/cir/search/upload` 存在
- 端点 `/api/cir/sknn/search/upload` 存在
- 端点 `/api/cir/status` 存在
- EncryptionInfo 字段正确

---

### core/cirtorch 前后端适配完成

已完成 **core/cirtorch 模块与前后端的完整适配**，实现了 CNN 图像检索和 SkNN 隐私保护检索的前端页面：

**后端修复**：
| 文件 | 变更 | 说明 |
|------|------|------|
| `backend/app/services/cir_service.py` | 重构 | 直接使用 `core/cirtorch`，移除对不存在模块的依赖 |
| `backend/app/routers/cir_retrieval.py` | 修复导入 | 修正 `SknnService` 导入路径 |
| `backend/app/routers/cir_retrieval.py` | 新增端点 | `/sknn/database/info` 和 `/sknn/database/load-demo` |

**前端新增**：
| 文件 | 变更 | 说明 |
|------|------|------|
| `frontend/src/app/cir/page.tsx` | 新建 | CNN 图像检索页面 |
| `frontend/src/components/Navbar.tsx` | 更新 | 添加 "CNN 检索" 导航链接 |

**工具脚本**：
| 文件 | 变更 | 说明 |
|------|------|------|
| `scripts/download_cir_model.py` | 新建 | 下载 CNN 检索预训练模型 |
| `scripts/build_cir_demo_db.py` | 新建 | 构建演示用加密检索数据库 |

**CIR 页面功能**：
- ✅ 服务状态监控（CIR 服务和 SkNN 状态）
- ✅ 检索模式切换（明文/隐私保护）
- ✅ 数据库管理（构建/加载）
- ✅ 图像上传检索
- ✅ SkNN 密钥生成

**使用方式**：
```bash
# 1. 下载预训练模型
python scripts/download_cir_model.py --model resnet101-gem

# 2. 构建演示数据库
python scripts/build_cir_demo_db.py --image-dir data/flickr-25k/images --save-dir data/cir_demo_db

# 3. 访问前端页面
# http://localhost:3000/cir
```

---

### 前后端检索与加密系统优化

已完成 **前后端检索与加密系统的全面优化**，实现真实 DCMH 模型集成和隐私保护检索：

**后端优化**：

| 模块 | 变更 | 功能 |
|------|------|------|
| DCMHService | 新增自动模型查找 | 自动加载最新训练的 DCMH 模型 |
| DatasetService | 新建服务 | 加载和管理 Flickr25K 数据集 |
| HashCacheService | 新建服务 | 预计算和缓存数据库哈希码 |
| Search API | 重构 | 使用真实模型和缓存哈希码检索 |
| Search Schema | 新增字段 | 加密状态信息、哈希码展示 |

**前端优化**：

| 组件 | 变更 | 功能 |
|------|------|------|
| Demo 页面 | 增强功能 | 图像上传检索、加密状态显示 |
| ResultsGrid | 新增功能 | 哈希码折叠展示 |
| ImageUpload | 保持功能 | 拖拽上传、实时预览 |

**新增功能**：
- ✅ 自动加载最新训练的 DCMH 模型（`results/flickr-25k/*/img_model.pth`）
- ✅ 哈希码预计算缓存（避免每次搜索重建数据库）
- ✅ 检索结果展示哈希码
- ✅ 加密状态可视化（查询加密、数据库加密指示器）
- ✅ 图像上传检索支持

**关键文件**：
- `backend/app/services/dcmh_service.py` - DCMH 模型服务（自动模型加载）
- `backend/app/services/dataset_service.py` - 数据集服务（新建）
- `backend/app/services/hash_cache_service.py` - 哈希缓存服务（新建）
- `backend/app/routers/search.py` - 搜索 API（重构）
- `frontend/src/app/demo/page.tsx` - 演示页面（增强）
- `frontend/src/components/ResultsGrid.tsx` - 结果展示（增强）

---

### 虚拟环境配置和 PyTorch GPU 安装

已添加完整的虚拟环境创建和 PyTorch GPU 版本安装指南：

**实施方案**：
| 步骤 | 命令 | 说明 |
|------|------|------|
| 创建虚拟环境 | `python -m venv .venv` | 使用 Python 3.13 |
| 激活虚拟环境 | `.venv\Scripts\activate` (Windows) | 隔离项目依赖 |
| 安装 PyTorch | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` | CUDA 12.4 支持 Python 3.13 |
| 安装依赖 | `pip install -r requirements.txt` | 安装其他项目依赖 |

**验证结果**：
- ✅ PyTorch 2.6.0+cu124 安装成功
- ✅ CUDA 12.4 可用
- ✅ GPU 识别成功（NVIDIA GeForce RTX 4060 Laptop GPU）

详见 [安装章节](#安装)。

---

### 训练优化（按需加载，内存占用 < 100MB）

DCMH 训练使用 Dataset 按需加载图像，内存占用极低：

| 优化措施 | 实现文件 | 效果 |
|----------|---------|------|
| **按需加载** | `training/dcmh_dataset.py` | 内存 < 100MB |
| **间隔验证** | `--valid_interval=10` 参数 | 每 N 个 epoch 验证，兼顾效率和监控 |

**使用方式**：

```bash
# 训练（每 10 个 epoch 验证一次，默认）
python training/train_dcmh.py train

# 训练（每 5 个 epoch 验证）
python training/train_dcmh.py train --valid_interval=5

# 训练（仅最终验证）
python training/train_dcmh.py train --valid_interval=0

# 训练完成后单独评估
python training/eval_dcmh.py --img_model_path=results/flickr-25k/时间戳/img_model.pth \
                             --txt_model_path=results/flickr-25k/时间戳/txt_model.pth

# 快捷方式：使用批处理/Shell 脚本（推荐）
# Windows
train_dcmh_low_memory.bat

# Linux/Mac
./train_dcmh_low_memory.sh
```

详见：
- `training/dcmh_dataset.py` - 按需加载 Dataset 类
- `training/eval_dcmh.py` - 训练后评估脚本
- `training/train_dcmh.py` - 训练脚本

---

### DCMH 原始实现对比分析和训练验证

已完成 **DCMH Reference 代码与当前项目实现的对比分析** 和 **小批量训练验证**：

**对比分析结果**：

| 模块 | Reference | 当前项目 | 状态 |
|------|-----------|---------|------|
| 配置参数 | `config.py` | `config/dcmh_config.py` | ✅ 完全一致 |
| 基础模块 | `BasicModule` | `DCMHBasicModule` | ✅ 功能一致 |
| 图像模块 | `ImgModule` | `DCMHImageModule` | ✅ 核心逻辑一致 |
| 文本模块 | `TxtModule` | `DCMHTextModule` | ✅ 核心逻辑一致 |
| 数据加载 | `data_handler.py` | `core/hashing/dcmh_data_loader.py` | ✅ 功能一致 |
| 评估指标 | `utils.py` | `core/retrieval/dcmh_metrics.py` | ✅ 完全一致 |
| 损失函数 | `main.py` 内联 | `training/train_dcmh.py` 内联 | ✅ 完全一致 |

**新增训练脚本**：
- `training/train_dcmh.py` - 基于 reference/DCMH/main.py 的完整训练脚本（损失函数内联）
- `training/train_dcmh_test.py` - 小批量训练测试脚本（快速验证）

**训练脚本使用方式**：
```bash
# 完整训练（使用 reference DCMH 训练流程）
python training/train_dcmh.py train --lr=0.01

# 小批量训练测试（快速验证实现正确性）
python training/train_dcmh_test.py

# 显示帮助
python training/train_dcmh.py help
```

**验证配置**（小批量测试）：
```python
training_size = 100    # 原 10000
query_size = 50        # 原 2000
database_size = 200    # 原 18015
batch_size = 16        # 原 128
max_epoch = 5          # 原 500
bit = 16               # 原 64
```

**验证结果**：
- ✅ 损失呈下降趋势（从 1.16 降至 0.89）
- ✅ mAP 在合理范围内（图像→文本约 0.45-0.47，文本→图像约 0.43-0.46）
- ✅ 训练验证了实现的正确性

详见 [training/train_dcmh.py](training/train_dcmh.py) 和 [training/train_dcmh_test.py](training/train_dcmh_test.py)。

---

### 精简说明

已删除以下冗余文件：
- `training/scheduler.py` - reference/DCMH 使用简单的线性衰减，不需要复杂调度器
- `training/dcmh_loss.py` - 损失函数已内联到 `train_dcmh.py` 中，与 reference 保持一致

---

### DCMH Reference 代码同步

已完成 **training 目录与 reference/DCMH 代码同步**：

- ✅ `training/train_flickr25k.py` - 完全复制 `reference/DCMH/main.py`
- ✅ `training/config.py` - 与 reference 保持一致
- ✅ `training/data_handler.py` - 与 reference 保持一致
- ✅ `training/utils.py` - 与 reference 保持一致

**运行方式**：
```bash
# 显示帮助
python training/train_flickr25k.py help

# 训练模型
python training/train_flickr25k.py train --lr=0.01

# 测试模型
python training/train_flickr25k.py test --load_img_path=result/img_model.pth
```

详见 [DCMH Reference 代码同步](#dcmh-reference-代码同步)

---

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
- ✅ **训练脚本**：损失函数内联到 `training/train_dcmh.py`，与 reference 一致

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
│   ├── train_dcmh.py       # DCMH 训练脚本（基于 reference/DCMH/main.py）
│   ├── train_dcmh_test.py  # DCMH 小批量训练测试脚本
│   └── __init__.py
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

### 创建虚拟环境（推荐）

```bash
# 克隆仓库
git clone <repository-url>
cd DeepHash-ASPE

# 创建虚拟环境（Python 3.13）
python -m venv .venv

# Windows - 激活虚拟环境
.venv\Scripts\activate

# Linux/Mac - 激活虚拟环境
source .venv/bin/activate
```

### 安装 PyTorch GPU 版本

**注意**：Python 3.13 需要使用 CUDA 12.4 或更高版本的 PyTorch。

```bash
# CUDA 12.4（推荐用于 Python 3.13）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1（适用于 Python 3.10-3.12）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU 版本（无 GPU）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 安装项目依赖

```bash
# 安装其他依赖
pip install -r requirements.txt

# 对于 MS-COCO 数据集（如需）
pip install pycocotools
```

### 验证 GPU 安装

```bash
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available())"
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

DCMH 的损失函数内联在 `training/train_dcmh.py` 训练脚本中，与 reference/DCMH/main.py 保持一致。

**损失组成**：
1. **对数损失**（基于全局相似性矩阵）：`-sum(S*theta - log(1+exp(theta)))`
2. **量化损失**（相对于动态 B 缓冲区）：`||B - F||² + ||B - G||²`
3. **平衡损失**（控制哈希码平衡性）：`||F.sum(dim=0)||² + ||G.sum(dim=0)||²`

```python
# 损失函数计算（来自 train_dcmh.py）
theta = torch.matmul(F, G.transpose(0, 1)) / 2
log_loss = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
quant_loss = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
balance_loss = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
total_loss = log_loss + gamma * quant_loss + eta * balance_loss
```

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

**完整功能（推荐）**：支持预训练权重、Resume、ASPE 评估、训练曲线

```bash
# 从头训练（使用预训练权重 + ASPE 评估 + 训练曲线）
python training/train_flickr25k.py \
    --data data/FLICKR-25K.mat \
    --pretrain data/imagenet-vgg-f.mat \
    --bit 64 --epochs 500 \
    --aspe-eval --plot

# 恢复训练（从中断点继续）
python training/train_flickr25k.py --resume results/flickr-25k/dcmh_flickr25k_latest.pth

# 仅 ASPE 评估（使用已有模型）
python training/train_flickr25k.py --eval-only --bit 64
```

**命令行参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `data/flickr25k/FLICKR-25K.mat` | 数据文件路径 |
| `--pretrain` | `data/imagenet-vgg-f.mat` | 预训练模型路径 |
| `--bit` | 64 | 哈希码维度 |
| `--batch-size` | 128 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--epochs` | 500 | 最大训练轮数 |
| `--resume` | - | 检查点路径（恢复训练） |
| `--aspe-eval` | - | 训练完成后执行 ASPE 评估 |
| `--plot` | - | 生成训练曲线图 |

**输出结果**（`results/flickr-25k/` 目录）：
- `dcmh_flickr25k_latest.pth` - 最新模型（可用于 resume）
- `dcmh_flickr25k_best.pth` - 最佳模型（验证 mAP 最高）
- `dcmh_flickr25k_checkpoint_*.pth` - 检查点（每 50 轮保存）
- `dcmh_training_results.json` - 训练历史
- `training_curves.png` - 训练曲线（综合图）
- `training_loss.png` - 损失曲线
- `training_map.png` - mAP 曲线
- `aspe_evaluation.json` - ASPE 评估结果

---

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
损失函数内联在 `training/train_dcmh.py` 中，包含：
- 对数损失（基于相似性矩阵）
- 量化损失（相对于动态 B 缓冲区）
- 平衡损失（控制哈希码平衡性）

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
