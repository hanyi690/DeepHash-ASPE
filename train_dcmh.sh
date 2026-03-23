#!/bin/bash
# DCMH 训练脚本（支持预训练权重、Resume、ASPE 评估）

echo ""
echo "================================================================"
echo "DCMH 训练脚本（Flickr25K 数据集）"
echo "================================================================"
echo ""

# 默认参数
DATA_PATH="data/FLICKR-25K.mat"
PRETRAIN_PATH="data/imagenet-vgg-f.mat"
BIT=64
EPOCHS=500
BATCH_SIZE=128
LR=1e-4
RESULT_DIR="results/flickr-25k"

# 显示帮助
if [ "$1" == "--help" ]; then
    echo "用法：./train_dcmh.sh [选项]"
    echo ""
    echo "选项:"
    echo "  --bit NUM         哈希码维度（默认：64）"
    echo "  --epochs NUM      训练轮数（默认：500）"
    echo "  --resume PATH     从检查点恢复训练"
    echo "  --no-pretrain     不使用预训练权重"
    echo "  --no-aspe         不执行 ASPE 评估"
    echo "  --no-plot         不生成训练曲线图"
    echo "  --help            显示帮助信息"
    echo ""
    exit 0
fi

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --bit)
            BIT="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --no-pretrain)
            PRETRAIN_PATH=""
            shift
            ;;
        --no-aspe)
            NO_ASPE=1
            shift
            ;;
        --no-plot)
            NO_PLOT=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# 构建命令
CMD="python training/train_flickr25k.py --data $DATA_PATH --bit $BIT --epochs $EPOCHS --batch-size $BATCH_SIZE --lr $LR --result-dir $RESULT_DIR"

# 添加预训练权重
if [ -n "$PRETRAIN_PATH" ]; then
    CMD="$CMD --pretrain $PRETRAIN_PATH"
fi

# 添加 resume 参数
if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

# 添加 ASPE 评估（默认执行，除非指定 --no-aspe）
if [ -z "$NO_ASPE" ]; then
    CMD="$CMD --aspe-eval"
fi

# 添加绘图（默认执行，除非指定 --no-plot）
if [ -z "$NO_PLOT" ]; then
    CMD="$CMD --plot"
fi

echo "执行命令：$CMD"
echo ""

# 执行训练
$CMD

echo ""
echo "================================================================"
echo "训练完成！"
echo "================================================================"
echo ""
echo "结果目录：$RESULT_DIR"
echo ""
