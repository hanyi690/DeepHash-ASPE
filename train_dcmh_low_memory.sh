#!/bin/bash
# DCMH 低内存训练脚本
# 适用于 7GB 内存限制环境

echo "============================================================"
echo "DCMH 低内存训练"
echo "============================================================"
echo ""
echo "选择训练模式:"
echo ""
echo "1. 低内存模式（按需加载图像，~100MB 内存）"
echo "   - 适合 7GB 内存限制"
echo "   - 训练速度中等"
echo ""
echo "2. 特征预提取模式（先提取特征，再训练）"
echo "   - 最低内存（~50MB）"
echo "   - 最快速度（训练速度提升 10 倍）"
echo "   - 需要额外时间提取特征"
echo ""
echo "3. 标准模式（加载所有数据到内存）"
echo "   - 需要 11+ GB 内存"
echo "   - 完整功能（训练中验证）"
echo ""
read -p "请输入选择 (1/2/3): " mode

case $mode in
    1)
        echo ""
        echo "正在启动低内存模式训练..."
        echo ""
        python training/train_dcmh.py train --low_memory=True --valid=False --max_epoch=500 --bit=64
        ;;
    2)
        echo ""
        echo "步骤 1: 提取特征..."
        echo ""
        python training/extract_features.py --data_path=./data/FLICKR-25K.mat

        if [ $? -ne 0 ]; then
            echo ""
            echo "特征提取失败，请检查数据文件是否存在"
            exit 1
        fi

        echo ""
        echo "步骤 2: 从特征训练..."
        echo ""
        # 获取最新提取的特征目录
        features_dir=$(ls -d data/features_* 2>/dev/null | sort -r | head -1)

        if [ -z "$features_dir" ]; then
            echo "未找到特征目录"
            exit 1
        fi

        echo "使用特征目录：$features_dir"
        python training/train_dcmh.py train --features_dir=$features_dir --valid=False --max_epoch=500 --bit=64
        ;;
    3)
        echo ""
        echo "正在启动标准模式训练..."
        echo ""
        echo "警告：此模式需要 11+ GB 内存"
        echo ""
        python training/train_dcmh.py train --max_epoch=500 --bit=64
        ;;
    *)
        echo "无效选择，请输入 1/2/3"
        exit 1
        ;;
esac

echo ""
echo "训练完成！"
echo "模型和结果保存在 results/flickr-25k/时间戳/ 目录"
echo ""
