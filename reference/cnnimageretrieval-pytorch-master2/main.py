import sys
import os
# 强制让 Python 把当前文件夹加入到搜索路径里
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from app.api.endpoints import router as api_router

# 确保目录存在
os.makedirs("app/data/retrieval_db", exist_ok=True)
os.makedirs("data/test/paris6k/jpg", exist_ok=True) 

app = FastAPI(
    title="Privacy-Preserving Image Retrieval API",
    description="基于 SkNN 和 GL18 的隐私保护图像检索后端"
)

# 允许跨域（极度重要！这是你本地电脑能连上云端 AutoDL 的关键）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载底库图片目录，让前端可以直接读取图片显示在网页上
app.mount("/data", StaticFiles(directory="data"), name="data")

# 包含咱们的检索路由
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "SkNN 隐私检索引擎已启动，随时待命！"}

if __name__ == "__main__":
    # AutoDL 暴露公网的默认端口是 6006
    uvicorn.run("main:app", host="0.0.0.0", port=6006, reload=True)