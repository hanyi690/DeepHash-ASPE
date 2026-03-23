# DCMH 笔记本式训练检索实现计划

**日期**: 2026-03-22
**需求**: 在 `training/train_flickr25k.py` 基础上添加 4 个功能

---

## 1. Context

### 现状

`training/train_flickr25k.py` 已有：
- ✅ 完整训练循环（500 行代码）
- ✅ Checkpoint 保存（模型、优化器、历史）
- ✅ 验证和 mAP 评估
- ✅ 训练结果 JSON 保存

`core/hashing/dcmh_model.py` 已支持：
- ✅ `DCMHModel(bit, y_dim, image_pretrain)` - 预训练权重参数

### 缺失功能

1. **预训练权重加载** - `DCMHTrainer._init_model()` 未传入 `pretrain_path`
2. **Resume 功能** - 无法从检查点恢复训练
3. **ASPE mAP 对比** - 训练完成后未自动执行
4. **训练曲线图** - 未生成可视化图表

---

## 2. 实现方案

### 2.1 预训练权重加载（最小改动）

**文件**: `training/train_flickr25k.py`

**改动**:

1. `__init__` 添加参数:
```python
def __init__(self, ..., pretrain_path: str = None, resume_path: str = None):
    self.pretrain_path = pretrain_path  # imagenet-vgg-f.mat 路径
    self.resume_path = resume_path      # 检查点路径
```

2. `_init_model` 加载预训练:
```python
def _init_model(self):
    self.model = DCMHModel(
        bit=self.bit_dim,
        y_dim=self.y_dim,
        image_pretrain=self.pretrain_path  # 新增
    )
```

### 2.2 Resume 功能

**新增方法**:
```python
def load_checkpoint(self, path):
    """从检查点恢复训练状态"""
    checkpoint = torch.load(path, map_location=self.device)

    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    self.best_map = checkpoint['best_map']
    self.best_epoch = checkpoint['best_epoch']

    # 恢复训练历史
    self.history = checkpoint.get('history', {
        'loss': [], 'map_i2t': [], 'map_t2i': [], 'lr': []
    })

    # 计算已训练轮数
    start_epoch = len(self.history['loss'])
    return start_epoch
```

**修改 `train` 方法**:
```python
def train(self, resume: bool = False):
    start_epoch = 0

    if resume or self.resume_path:
        path = self.resume_path or (self.result_dir / 'dcmh_flickr25k_latest.pth')
        start_epoch = self.load_checkpoint(path)
        print(f"从第 {start_epoch} 轮恢复训练")

    for epoch in range(start_epoch, self.max_epoch):
        ...
```

### 2.3 ASPE mAP 对比

**新增方法**:
```python
def evaluate_aspe(self):
    """ASPE 加密前后 mAP 对比评估"""
    from core.aspe.dcmh_wrapper import ASPEForDCMH

    # 生成哈希码
    img_codes, txt_codes = self.generate_hash_codes(
        self.database_images, self.database_tags
    )
    query_img_codes, query_txt_codes = self.generate_hash_codes(
        self.query_images, self.query_tags
    )

    # ASPE 加密
    aspe = ASPEForDCMH(bit_dim=self.bit_dim, seed=42)
    enc_db = aspe.GenEnc(img_codes)
    enc_query = aspe.GenTrap(query_img_codes)

    # 密文 mAP
    ciphertext_map = aspe.calc_ciphertext_map(
        enc_query, enc_db,
        self.query_labels.numpy(),
        self.database_labels.numpy()
    )

    # 明文 mAP（使用参考实现）
    plaintext_map = calc_map_k(
        torch.sign(torch.from_numpy(query_img_codes)),
        torch.sign(torch.from_numpy(img_codes)),
        self.query_labels, self.database_labels
    )

    # 打印对比
    print(f"\nASPE mAP 对比:")
    print(f"  明文 mAP: {plaintext_map:.6f}")
    print(f"  密文 mAP: {ciphertext_map:.6f}")
    print(f"  差异：{abs(plaintext_map - ciphertext_map):.8f}")

    return {
        'plaintext_map': float(plaintext_map.item()),
        'ciphertext_map': ciphertext_map,
        'map_diff': abs(plaintext_map.item() - ciphertext_map)
    }
```

### 2.4 训练曲线图

**新增方法**:
```python
def plot_training_curves(self):
    """生成训练曲线图"""
    import matplotlib.pyplot as plt

    # 训练损失曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(self.history['loss'], 'b-', label='Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curve')
    ax.legend()
    plt.savefig(self.result_dir / 'training_loss.png')
    plt.close()

    # mAP 曲线
    if self.history['map_i2t']:
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(self.history['map_i2t']) + 1)
        ax.plot(epochs, self.history['map_i2t'], 'b-', label='Image→Text')
        ax.plot(epochs, self.history['map_t2i'], 'r-', label='Text→Image')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.set_title('Retrieval mAP')
        ax.legend()
        plt.savefig(self.result_dir / 'training_map.png')
        plt.close()
```

---

## 3. 命令行参数更新

**修改 `__main__` 部分**:

```python
parser.add_argument('--pretrain', type=str,
                   default='data/imagenet-vgg-f.mat',
                   help='预训练模型路径')
parser.add_argument('--resume', type=str, default=None,
                   help='从检查点恢复训练')
parser.add_argument('--aspe-eval', action='store_true',
                   help='训练完成后执行 ASPE 评估')
parser.add_argument('--plot', action='store_true',
                   help='生成训练曲线图')

# 使用
trainer, history = train_flickr25k(
    ...,
    pretrain_path=args.pretrain,
    resume_path=args.resume,
    ...
)

# 训练后评估
if args.aspe_eval:
    trainer.evaluate_aspe()

if args.plot:
    trainer.plot_training_curves()
```

---

## 4. 使用示例

```bash
# 从头训练（使用预训练权重）
python training/train_flickr25k.py \
    --bit 64 --epochs 500 \
    --pretrain data/imagenet-vgg-f.mat \
    --aspe-eval --plot

# 恢复训练
python training/train_flickr25k.py --resume results/flickr-25k/dcmh_flickr25k_latest.pth

# 仅 ASPE 评估
python training/train_flickr25k.py --eval-only
```

---

## 5. 输出结果

```
results/flickr-25k/
├── dcmh_flickr25k_latest.pth       # 最新模型（可 resume）
├── dcmh_flickr25k_best.pth         # 最佳模型
├── dcmh_flickr25k_checkpoint_*.pth # 检查点（每 50 轮）
├── dcmh_training_results.json      # 训练历史
├── training_loss.png               # 损失曲线
├── training_map.png                # mAP 曲线
└── aspe_evaluation.json            # ASPE 评估结果
```

---

## 6. 验证

训练完成后验证：

```python
# 1. 检查点存在
assert Path('results/flickr-25k/dcmh_flickr25k_latest.pth').exists()

# 2. 训练历史完整
with open('results/flickr-25k/dcmh_training_results.json') as f:
    data = json.load(f)
assert len(data['history']['loss']) == 500

# 3. 图表生成
assert Path('results/flickr-25k/training_loss.png').exists()
assert Path('results/flickr-25k/training_map.png').exists()

# 4. ASPE mAP 差异 < 1e-6
assert data['aspe']['map_diff'] < 1e-6
```

---

## 7. 改动统计

| 文件 | 新增行数 | 修改行数 |
|------|----------|----------|
| `training/train_flickr25k.py` | ~120 行 | ~20 行 |

**总计**: ~140 行改动
