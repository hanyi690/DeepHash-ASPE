"""
FastAPI 主应用

DCMH + ASPE 隐私保护跨模态检索演示系统后端
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 导入路由
from app.routers import images, texts, hash_codes, encrypt, search, metrics

# 创建 FastAPI 应用
app = FastAPI(
    title="DCMH + ASPE 演示系统",
    description="基于深度跨模态哈希和非对称标量积保持加密的隐私保护跨模态检索演示系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建静态文件目录
static_dir = Path(__file__).parent.parent.parent / "backend" / "app" / "static"
static_dir.mkdir(exist_ok=True)

# 挂载静态文件
if (static_dir / "uploads").exists() or True:
    app.mount("/uploads", StaticFiles(directory=str(static_dir / "uploads"), html=True), mount_path="/uploads")

# 注册路由
app.include_router(images.router)
app.include_router(texts.router)
app.include_router(hash_codes.router)
app.include_router(encrypt.router)
app.include_router(search.router)
app.include_router(metrics.router)


@app.get("/")
async def root():
    """根端点。"""
    return {
        "message": "欢迎使用 DCMH + ASPE 演示系统 API",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/api/health")
async def health_check():
    """健康检查。"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


@app.on_event("startup")
async def startup_event():
    """应用启动时执行。"""
    print("=" * 50)
    print("DCMH + ASPE 演示系统启动")
    print("=" * 50)

    # 初始化服务
    try:
        from app.services.dcmh_service import get_dcmh_service
        from app.services.aspe_service import get_aspe_service
        from app.services.coco_service import get_coco_service

        # 初始化 DCMH 服务
        print("正在初始化 DCMH 服务...")
        dcmh_service = get_dcmh_service(bit_dim=64)
        print(f"  - 哈希码维度：{dcmh_service.bit_dim}")

        # 初始化 ASPE 服务
        print("正在初始化 ASPE 服务...")
        aspe_service = get_aspe_service(bit_dim=64, seed=42)
        print(f"  - ASPE 密钥种子：{aspe_service.seed}")

        # 初始化 COCO 服务
        print("正在初始化 COCO 服务...")
        coco_service = get_coco_service()
        print(f"  - 数据集已加载：{coco_service.images_data is not None}")

        print("=" * 50)
        print("服务初始化完成")
        print("=" * 50)

    except Exception as e:
        print(f"服务初始化警告：{e}")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行。"""
    print("DCMH + ASPE 演示系统关闭")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
