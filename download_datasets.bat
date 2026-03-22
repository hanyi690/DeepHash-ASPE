@echo off
chcp 65001 >nul
REM DCMH + ASPE 数据集下载脚本
REM 用于下载系统测试所需的数据集

echo ============================================================
echo DCMH + ASPE 数据集下载工具
echo ============================================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

echo [信息] 检查依赖库...
python -c "import numpy, scipy, h5py, requests, tqdm, PIL" >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 缺少必要的依赖库
    echo [信息] 正在安装依赖库...
    pip install numpy scipy h5py requests tqdm pillow
)

echo.
echo ============================================================
echo 请选择要下载的数据集:
echo ============================================================
echo   1. Flickr-25K (推荐 - 参考实现原生支持)
echo   2. IAPR TC-12 (多标签基准)
echo   3. NUS-WIDE (学术标准基准)
echo   4. 全部下载
echo   0. 退出
echo.

set /p choice=请输入选项 (0-4):

if "%choice%"=="1" (
    goto :download_flickr
) else if "%choice%"=="2" (
    goto :download_iapr
) else if "%choice%"=="3" (
    goto :download_nus
) else if "%choice%"=="4" (
    goto :download_all
) else if "%choice%"=="0" (
    exit /b 0
) else (
    echo [错误] 无效选项
    pause
    exit /b 1
)

:download_flickr
echo.
echo ============================================================
echo Flickr-25K 数据集下载
echo ============================================================
echo.
echo Flickr-25K 数据集信息:
echo   - 图像数量：25,015
echo   - 类别数量：24
echo   - 数据格式：.mat 文件
echo.
echo 下载方式:
echo   1. 手动下载（推荐）
echo   2. 自动下载（如果可用）
echo.
set /p download_method=请选择下载方式 (1-2):

if "%download_method%"=="1" (
    echo.
    echo 请访问以下链接下载:
    echo   百度网盘：https://pan.baidu.com/s/1X5BTyux524aUyqHpFGPPlA
    echo   提取码：eico
    echo.
    echo 下载后请将 FLICKR-25K.mat 放置到:
    echo   %CD%\data\flickr25k\FLICKR-25K.mat
    echo.
    pause

    REM 验证文件
    if exist "data\flickr25k\FLICKR-25K.mat" (
        echo [成功] 文件已就绪
        python -c "from data.dataset_downloader import Flickr25KDownloader; d = Flickr25KDownloader('./data/flickr25k'); d.verify_dataset()"
    ) else (
        echo [警告] 文件未找到，请确认文件路径
    )
) else (
    echo 自动下载功能需要配置实际 URL
    python data\dataset_downloader.py --dataset flickr25k
)

goto :menu

:download_iapr
echo.
echo ============================================================
echo IAPR TC-12 数据集下载
echo ============================================================
echo.
echo IAPR TC-12 数据集信息:
echo   - 图像数量：22,127
echo   - 类别数量：255
echo   - 数据格式：JPEG + .mat/.txt 标签
echo.
echo 请访问 ImageCLEF 官网下载:
echo   http://www.imageclef.org/photodata
echo.
echo 需要注册并申请下载权限
echo.
pause

REM 验证文件
if exist "data\iapr_tc12\IAPR-TC12.mat" (
    echo [成功] 文件已就绪
    python -c "from data.dataset_downloader import IAPRTC12Downloader; d = IAPRTC12Downloader('./data/iapr_tc12'); d.verify_dataset()"
) else (
    echo [信息] 文件未找到，请手动下载后运行验证
)

goto :menu

:download_nus
echo.
echo ============================================================
echo NUS-WIDE 数据集下载
echo ============================================================
echo.
echo NUS-WIDE 数据集信息:
echo   - 图像数量：269,648
echo   - 类别数量：81
echo   - 数据格式：URL 列表 + GroundTruth 文件
echo.
echo 请访问 NUS 官网下载:
echo   http://lms.comp.nus.edu.sg/research/nuswide.shtml
echo.
echo 下载说明:
echo   1. 下载 ImageList.txt（图像 URL 列表）
echo   2. 下载 Groundtruth.zip 或 Concepts81.zip（标签文件）
echo   3. 运行图像处理脚本从 URL 下载图像
echo.
pause

REM 运行下载脚本
python data\nuswide_downloader.py --dataset nuswide

goto :menu

:download_all
echo.
echo ============================================================
echo 下载全部数据集
echo ============================================================
echo.
python data\dataset_downloader.py --dataset all
goto :menu

:menu
echo.
echo ============================================================
echo 是否返回主菜单？
echo ============================================================
set /p continue=是否继续？(Y/N):
if /i "%continue%"=="Y" (
    goto :start
)

:start
echo.
echo ============================================================
echo DCMH + ASPE 数据集下载工具
echo ============================================================
echo.
echo 数据集下载说明:
echo.
echo 1. Flickr-25K (推荐)
echo    - 百度网盘：https://pan.baidu.com/s/1X5BTyux524aUyqHpFGPPlA
echo    - 提取码：eico
echo    - 目标路径：data\flickr25k\FLICKR-25K.mat
echo.
echo 2. IAPR TC-12
echo    - 官网：http://www.imageclef.org/photodata
echo    - 需要注册申请
echo    - 目标路径：data\iapr_tc12\IAPR-TC12.mat
echo.
echo 3. NUS-WIDE
echo    - 官网：http://lms.comp.nus.edu.sg/research/nuswide.shtml
echo    - 目标路径：data\nuswide\NUS-WIDE.mat
echo.
echo ============================================================

pause
