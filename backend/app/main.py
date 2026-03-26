"""
FastAPI 主应用

DCMH + ASPE 隐私保护跨模态检索演示系统后端
支持多数据集切换和 GPU 自动检测
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import sys
import torch

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 导入路由
from app.routers import images, texts, hash_codes, encrypt, search, metrics, cir_retrieval, tags, datasets

# 导入数据集配置
from config.dataset_config import DATA_ROOT

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
    app.mount("/uploads", StaticFiles(directory=str(static_dir / "uploads"), html=True))

# 挂载 CIR 数据集图像目录（用于图搜图显示结果图像）
cir_test_dir = DATA_ROOT / "test"
if cir_test_dir.exists():
    app.mount("/cir-images", StaticFiles(directory=str(cir_test_dir), html=True), name="cir-images")

# 挂载 Flickr25K 图像目录（用于跨模态检索显示结果图像）
flickr_images_dir = DATA_ROOT / "dcmh" / "flickr25k" / "mirflickr"
if flickr_images_dir.exists():
    app.mount("/flickr-images", StaticFiles(directory=str(flickr_images_dir), html=True), name="flickr-images")

# 注册路由
app.include_router(images.router)
app.include_router(texts.router)
app.include_router(hash_codes.router)
app.include_router(encrypt.router)
app.include_router(search.router)
app.include_router(metrics.router)
app.include_router(cir_retrieval.router)
app.include_router(tags.router)
app.include_router(datasets.router)


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

    # GPU 检测
    if torch.cuda.is_available():
        print(f"GPU 可用：{torch.cuda.get_device_name(0)}")
        print(f"GPU 内存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("GPU 不可用，将使用 CPU 运行")

    # 初始化服务
    try:
        from app.services.dcmh_service import get_dcmh_service
        from app.services.aspe_service import get_aspe_service
        from app.services.dataset_service import get_dataset_service

        # 初始化 DCMH 服务
        print("正在初始化 DCMH 服务...")
        dcmh_service = get_dcmh_service(bit_dim=64)
        print(f"  - 哈希码维度：{dcmh_service.bit_dim}")
        print(f"  - 运行设备：{dcmh_service.device}")

        # 初始化 ASPE 服务
        print("正在初始化 ASPE 服务...")
        aspe_service = get_aspe_service(bit_dim=64, seed=42)
        print(f"  - ASPE 密钥种子：{aspe_service.seed}")

        # 初始化数据集服务
        print("正在初始化数据集服务...")
        dataset_service = get_dataset_service()
        print(f"  - 默认数据路径：{dataset_service.data_path}")

        print("=" * 50)
        print("服务初始化完成")
        print("=" * 50)

    except Exception as e:
        print(f"服务初始化警告：{e}")

    # 初始化 CIR 服务（可选，懒加载）
    try:
        print("正在初始化 CIR 服务...")
        from app.services.cir_service import get_cir_service
        from config.dataset_config import check_cir_dataset_exists, CIR_DATASET_CONFIGS

        cir_service = get_cir_service()
        print(f"  - CIR 服务已创建")

        # 检查可用数据集
        available_datasets = []
        for dataset_name in CIR_DATASET_CONFIGS.keys():
            exists, path, message = check_cir_dataset_exists(dataset_name)
            if exists:
                available_datasets.append(dataset_name)
                print(f"  - {dataset_name}: 已就绪")
            else:
                print(f"  - {dataset_name}: {message}")

        if available_datasets:
            print(f"  可用数据集: {', '.join(available_datasets)}")
            print("  提示: CIR 数据将在首次请求时加载（懒加载模式）")
        else:
            print("  警告: 没有可用的 CIR 数据集")
            print("  请先下载数据集或运行缓存构建脚本:")
            print("    python scripts/build_all_cache.py --type cir")

    except Exception as e:
        print(f"CIR 服务初始化警告：{e}")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行。"""
    print("DCMH + ASPE 演示系统关闭")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
