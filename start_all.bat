@echo off
echo ============================================
echo   DeepHash-ASPE Demo - Quick Start
echo ============================================
echo.

REM Check Python
echo [1/3] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python.
    pause
    exit /b 1
)
echo [OK] Python found

REM Start backend in new window
echo [2/3] Starting backend server...
start "DeepHash-ASPE Backend" cmd /k "cd /d %~dp0backend && uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo [OK] Backend started (port 8000)

REM Check Node.js
echo [3/3] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo [WARN] Node.js not found. Frontend will not start.
    echo Please install Node.js to run the frontend.
    pause
    exit /b 1
)

REM Start frontend in new window
cd /d %~dp0frontend
echo [OK] Node.js found
echo Starting frontend...
start "DeepHash-ASPE Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"
echo [OK] Frontend started (port 3000)

echo.
echo ============================================
echo   Services Started Successfully!
echo ============================================
echo.
echo   Backend API:  http://localhost:8000
echo   API Docs:     http://localhost:8000/docs
echo   Frontend:     http://localhost:3000
echo.
echo   Press any key to close this window (services keep running)
echo ============================================
pause >nul
