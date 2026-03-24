@echo off
REM DCMH 训练脚本
REM 使用 Dataset 按需加载数据，内存占用 < 100MB

echo ============================================================
echo DCMH 训练
echo ============================================================
echo.
echo 选择训练模式:
echo.
echo 1. 低内存模式（按需加载图像，~100MB 内存）
echo    - 适合 7GB 内存限制
echo    - 训练速度中等
echo.
echo 2. 特征预提取模式（先提取特征，再训练）
echo    - 最低内存（~50MB）
echo    - 最快速度（训练速度提升 10 倍）
echo    - 需要额外时间提取特征
echo.
set /p mode="请输入选择 (1/2): "

if "%mode%"=="1" (
    echo.
    echo 正在启动训练...
    echo.
    python training/train_dcmh.py train --valid=False --max_epoch=500 --bit=64
    goto :end
)

if "%mode%"=="2" (
    echo.
    echo 步骤 1: 提取特征...
    echo.
    python training/extract_features.py --data_path=./data/FLICKR-25K.mat

    if errorlevel 1 (
        echo.
        echo 特征提取失败，请检查数据文件是否存在
        pause
        exit /b 1
    )

    echo.
    echo 步骤 2: 从特征训练...
    echo.
    REM 获取最新提取的特征目录
    for /f "delims=" %%i in ('dir /b /o-d data\features_* 2^>nul ^| findstr /n "^" ^| findstr "^1:"') do set features_dir=%%~nxi
    set features_dir=%features_dir:~2%

    if not defined features_dir (
        echo 未找到特征目录
        pause
        exit /b 1
    )

    echo 使用特征目录：data\%features_dir%
    python training/train_dcmh.py train --features_dir=./data/%features_dir% --valid=False --max_epoch=500 --bit=64
    goto :end
)

echo 无效选择，请输入 1 或 2
:end
echo.
echo 训练完成！
echo 模型和结果保存在 results\flickr-25k\时间戳\ 目录
echo.
pause