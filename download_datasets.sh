#!/bin/bash
# DCMH + ASPE 数据集下载脚本 (Linux/Mac)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "DCMH + ASPE 数据集下载工具"
echo "============================================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 Python3，请先安装 Python 3.8+"
    exit 1
fi

# 检查依赖
echo "[信息] 检查依赖库..."
if ! python3 -c "import numpy, scipy, h5py, requests, tqdm, PIL" 2>/dev/null; then
    echo "[警告] 缺少必要的依赖库"
    echo "[信息] 正在安装依赖库..."
    pip3 install numpy scipy h5py requests tqdm pillow
fi

echo ""
echo "============================================================"
echo "请选择要下载的数据集:"
echo "============================================================"
echo "  1. Flickr-25K (推荐 - 参考实现原生支持)"
echo "  2. IAPR TC-12 (多标签基准)"
echo "  3. NUS-WIDE (学术标准基准)"
echo "  4. 全部下载"
echo "  0. 退出"
echo ""

read -p "请输入选项 (0-4): " choice

case $choice in
    1)
        echo ""
        echo "============================================================"
        echo "Flickr-25K 数据集下载"
        echo "============================================================"
        echo ""
        echo "Flickr-25K 数据集信息:"
        echo "  - 图像数量：25,015"
        echo "  - 类别数量：24"
        echo "  - 数据格式：.mat 文件"
        echo ""
        echo "下载方式:"
        echo "  1. 手动下载（推荐）"
        echo "  2. 自动下载（如果可用）"
        echo ""
        read -p "请选择下载方式 (1-2): " method

        if [ "$method" = "1" ]; then
            echo ""
            echo "请访问以下链接下载:"
            echo "  百度网盘：https://pan.baidu.com/s/1X5BTyux524aUyqHpFGPPlA"
            echo "  提取码：eico"
            echo ""
            echo "下载后请将 FLICKR-25K.mat 放置到:"
            echo "  $(pwd)/data/flickr25k/FLICKR-25K.mat"
            echo ""
            read -p "按回车继续..."

            if [ -f "data/flickr25k/FLICKR-25K.mat" ]; then
                echo "[成功] 文件已就绪"
                python3 -c "from data.dataset_downloader import Flickr25KDownloader; d = Flickr25KDownloader('./data/flickr25k'); d.verify_dataset()"
            else
                echo "[警告] 文件未找到，请确认文件路径"
            fi
        else
            echo "自动下载功能需要配置实际 URL"
            python3 data/dataset_downloader.py --dataset flickr25k
        fi
        ;;

    2)
        echo ""
        echo "============================================================"
        echo "IAPR TC-12 数据集下载"
        echo "============================================================"
        echo ""
        echo "IAPR TC-12 数据集信息:"
        echo "  - 图像数量：22,127"
        echo "  - 类别数量：255"
        echo "  - 数据格式：JPEG + .mat/.txt 标签"
        echo ""
        echo "请访问 ImageCLEF 官网下载:"
        echo "  http://www.imageclef.org/photodata"
        echo ""
        echo "需要注册并申请下载权限"
        echo ""
        read -p "按回车继续..."

        if [ -f "data/iapr_tc12/IAPR-TC12.mat" ]; then
            echo "[成功] 文件已就绪"
            python3 -c "from data.dataset_downloader import IAPRTC12Downloader; d = IAPRTC12Downloader('./data/iapr_tc12'); d.verify_dataset()"
        else
            echo "[信息] 文件未找到，请手动下载后运行验证"
        fi
        ;;

    3)
        echo ""
        echo "============================================================"
        echo "NUS-WIDE 数据集下载"
        echo "============================================================"
        echo ""
        echo "NUS-WIDE 数据集信息:"
        echo "  - 图像数量：269,648"
        echo "  - 类别数量：81"
        echo "  - 数据格式：URL 列表 + GroundTruth 文件"
        echo ""
        echo "请访问 NUS 官网下载:"
        echo "  http://lms.comp.nus.edu.sg/research/nuswide.shtml"
        echo ""
        echo "下载说明:"
        echo "  1. 下载 ImageList.txt（图像 URL 列表）"
        echo "  2. 下载 Groundtruth.zip 或 Concepts81.zip（标签文件）"
        echo "  3. 运行图像处理脚本从 URL 下载图像"
        echo ""
        read -p "按回车继续..."

        python3 data/nuswide_downloader.py --dataset nuswide
        ;;

    4)
        echo ""
        echo "============================================================"
        echo "下载全部数据集"
        echo "============================================================"
        echo ""
        python3 data/dataset_downloader.py --dataset all
        ;;

    0)
        exit 0
        ;;

    *)
        echo "[错误] 无效选项"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "数据集下载说明:"
echo "============================================================"
echo ""
echo "1. Flickr-25K (推荐)"
echo "   - 百度网盘：https://pan.baidu.com/s/1X5BTyux524aUyqHpFGPPlA"
echo "   - 提取码：eico"
echo "   - 目标路径：data/flickr25k/FLICKR-25K.mat"
echo ""
echo "2. IAPR TC-12"
echo "   - 官网：http://www.imageclef.org/photodata"
echo "   - 需要注册申请"
echo "   - 目标路径：data/iapr_tc12/IAPR-TC12.mat"
echo ""
echo "3. NUS-WIDE"
echo "   - 官网：http://lms.comp.nus.edu.sg/research/nuswide.shtml"
echo "   - 目标路径：data/nuswide/NUS-WIDE.mat"
echo ""
echo "============================================================"
