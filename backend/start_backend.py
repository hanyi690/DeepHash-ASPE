"""
后端启动脚本

用法：
    python start_backend.py
"""

import sys
from pathlib import Path

# 获取项目根目录 (backend 的父目录)
PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_DIR = Path(__file__).parent

# 添加项目根目录到 sys.path
sys.path.insert(0, str(PROJECT_ROOT))
# 添加 references/DCMH 目录
sys.path.insert(0, str(PROJECT_ROOT / "references" / "DCMH"))

import uvicorn

if __name__ == "__main__":
    from app.main import app
    print("=" * 50)
    print("DCMH + ASPE 演示系统 - 后端启动")
    print("=" * 50)
    print("API 文档：http://localhost:8000/docs")
    print("Redoc: http://localhost:8000/redoc")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)