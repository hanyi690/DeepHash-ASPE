@echo off
REM DCMH 标准模式训练脚本

echo.
echo ================================================================
echo DCMH 训练（标准模式 + GPU）
echo ================================================================
echo.

REM 激活虚拟环境
call .venv\Scripts\activate

REM 运行标准模式训练
python training/train_dcmh.py train --low_memory=False

pause