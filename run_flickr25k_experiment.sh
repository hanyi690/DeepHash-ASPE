#!/bin/bash
# =============================================================================
# Flickr25K 完整系统测试和实验启动脚本 (Linux/Mac)
# =============================================================================
# 使用 Flickr25K 数据集完成 DCMH + ASPE 系统的全部测试和实验
# 结果输出到 results/flickr-25k 目录
# =============================================================================

set -e

echo "============================================================================"
echo "Flickr25K 完整系统测试和实验"
echo "============================================================================"
echo ""

# 参数配置
DATA_PATH="data/flickr25k/FLICKR-25K.mat"
BIT_DIM=64
MAX_EPOCH=500
BATCH_SIZE=128
LEARNING_RATE=0.0001
RESULT_DIR="results/flickr-25k"

echo "参数配置:"
echo "  数据路径：$DATA_PATH"
echo "  哈希维度：$BIT_DIM bits"
echo "  训练轮数：$MAX_EPOCH"
echo "  批次大小：$BATCH_SIZE"
echo "  学习率：$LEARNING_RATE"
echo "  结果目录：$RESULT_DIR"
echo ""

# 检查数据文件
if [ ! -f "$DATA_PATH" ]; then
    echo "[错误] 数据文件不存在：$DATA_PATH"
    echo ""
    echo "请先下载 FLICKR-25K.mat 文件到 data/flickr25k/ 目录"
    echo "下载地址：https://pan.baidu.com/s/1X5BTyux524aUyqHpFGPPlA"
    echo "提取码：eico"
    echo ""
    exit 1
fi

echo "[✓] 数据文件检查通过"
echo ""

# 创建结果目录
mkdir -p "$RESULT_DIR"

echo "============================================================================"
echo "开始运行实验..."
echo "============================================================================"
echo ""

# 运行实验脚本
python experiments/run_flickr25k.py \
    --data "$DATA_PATH" \
    --bit $BIT_DIM \
    --epochs $MAX_EPOCH \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --result-dir "$RESULT_DIR"

echo ""
echo "============================================================================"
echo "实验完成！"
echo "============================================================================"
echo ""
echo "结果文件:"
echo "  - $RESULT_DIR/flickr25k_experiment_report.md (实验报告)"
echo "  - $RESULT_DIR/flickr25k_experiment_results.json (详细数据)"
echo "  - $RESULT_DIR/*.png (可视化图表)"
echo "  - $RESULT_DIR/dcmh_best.pth (最佳模型)"
echo ""
