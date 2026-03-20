# DCMH + ASPE 演示系统 - 文件清单

## 后端 (FastAPI)

### 核心文件
- `backend/app/main.py` - FastAPI 主应用入口
- `backend/requirements.txt` - Python 依赖
- `backend/start_backend.py` - 启动脚本

### 服务层 (services/)
- `backend/app/services/dcmh_service.py` - DCMH 哈希码生成服务
- `backend/app/services/aspe_service.py` - ASPE 加密服务
- `backend/app/services/coco_service.py` - MS-COCO 数据集服务

### API 路由 (routers/)
- `backend/app/routers/images.py` - 图像 API
- `backend/app/routers/texts.py` - 文本 API
- `backend/app/routers/hash_codes.py` - 哈希码 API
- `backend/app/routers/encrypt.py` - 加密 API
- `backend/app/routers/search.py` - 检索 API
- `backend/app/routers/metrics.py` - 指标 API

### 数据模式 (schemas/)
- `backend/app/schemas/search.py` - Pydantic 数据模式

### 包初始化
- `backend/app/__init__.py`
- `backend/app/routers/__init__.py`
- `backend/app/services/__init__.py`
- `backend/app/schemas/__init__.py`

---

## 前端 (Next.js 14)

### 配置文件
- `frontend/package.json` - NPM 依赖
- `frontend/next.config.js` - Next.js 配置
- `frontend/tailwind.config.js` - Tailwind CSS 配置
- `frontend/tsconfig.json` - TypeScript 配置
- `frontend/postcss.config.js` - PostCSS 配置
- `frontend/.env.local` - 环境变量

### 页面 (src/app/)
- `frontend/src/app/layout.tsx` - 根布局
- `frontend/src/app/globals.css` - 全局样式
- `frontend/src/app/page.tsx` - 首页
- `frontend/src/app/demo/page.tsx` - 演示页面
- `frontend/src/app/dataset/page.tsx` - 数据集页面
- `frontend/src/app/encrypt/page.tsx` - 加密可视化页面
- `frontend/src/app/metrics/page.tsx` - 指标页面

### 组件 (src/components/)
- `frontend/src/components/ImageUpload.tsx` - 图像上传组件
- `frontend/src/components/ResultsGrid.tsx` - 搜索结果网格
- `frontend/src/components/MetricsChart.tsx` - 指标图表组件
- `frontend/src/components/Navbar.tsx` - 导航栏组件

### 工具库 (src/lib/)
- `frontend/src/lib/api.ts` - API 客户端

---

## 文档
- `README.md` - 项目主文档（已更新演示系统说明）
- `STARTUP.md` - 快速启动指南

---

## 总计
- 后端文件：14 个
- 前端文件：14 个
- 文档文件：2 个
- **总计：30 个新文件**

---

## 验收状态

✅ 前端 5 个页面：首页、演示、数据集、加密可视化、指标
✅ 后端 API 端点：images, texts, hash, encrypt, search, metrics
✅ 服务层：DCMHService, ASPEService, COCOService
✅ 数据模式：完整的 Pydantic 模式定义
✅ 组件：ImageUpload, ResultsGrid, MetricsChart, Navbar
✅ API 客户端：封装所有 API 调用
✅ 文档：README 更新，启动指南

---

## 启动说明

### 后端
```bash
cd backend
pip install -r requirements.txt
python start_backend.py
# 或 uvicorn app.main:app --reload
```

### 前端
```bash
cd frontend
npm install
npm run dev
```

### 访问
- 前端：http://localhost:3000
- API 文档：http://localhost:8000/docs
