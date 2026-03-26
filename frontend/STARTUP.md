# DCMH + ASPE 演示系统

## 快速启动

### 启动后端

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 启动前端

```bash
cd frontend
npm install
npm run dev
```

### 访问

- 前端：http://localhost:3000
- API 文档：http://localhost:8000/docs
