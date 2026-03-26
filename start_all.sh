#!/bin/bash

echo "============================================"
echo "  DeepHash-ASPE 演示系统 - 一键启动"
echo "============================================"
echo ""

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 检查 Python
echo "[1/4] 检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到 Python3，请先安装 Python3"
    exit 1
fi
echo "[✓] Python 已安装：$(python3 --version)"

# 激活虚拟环境
echo "[2/4] 激活虚拟环境..."
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    echo "[✓] 虚拟环境已激活"
else
    echo "[警告] 未找到虚拟环境，使用系统 Python"
fi

# 检查 Node.js
echo "[3/4] 检查 Node.js 环境..."
if ! command -v node &> /dev/null; then
    echo "[警告] 未找到 Node.js，前端将无法启动"
fi

# 启动后端（后台）
echo "[4/4] 启动后端服务..."
cd "$SCRIPT_DIR/backend"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "[✓] 后端已启动（PID: $BACKEND_PID，端口 8000）"

sleep 2

# 启动前端（后台）
cd "$SCRIPT_DIR/frontend"
if command -v node &> /dev/null; then
    npm run dev &
    FRONTEND_PID=$!
    echo "[✓] 前端已启动（PID: $FRONTEND_PID，端口 3000）"
fi

echo ""
echo "============================================"
echo "  启动完成!"
echo "============================================"
echo ""
echo "  后端 API:   http://localhost:8000"
echo "  API 文档：  http://localhost:8000/docs"
if command -v node &> /dev/null; then
    echo "  前端界面：http://localhost:3000"
fi
echo ""
echo "  按 Ctrl+C 停止所有服务"
echo "============================================"

# 等待用户中断
wait