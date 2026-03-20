"""
后端启动脚本

用法：
    python start_backend.py
"""

import sys
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "reference" / "DCMH"))

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
