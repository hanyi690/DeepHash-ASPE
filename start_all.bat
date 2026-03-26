@echo off
echo ============================================
echo   DeepHash-ASPE Demo - Quick Start
echo ============================================
echo.

REM Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Check Python
echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python.
    pause
    exit /b 1
)
echo [OK] Python found

REM Activate virtual environment
echo [2/4] Activating virtual environment...
if exist "%SCRIPT_DIR%.venv\Scripts\activate.bat" (
    call "%SCRIPT_DIR%.venv\Scripts\activate.bat"
    echo [OK] Virtual environment activated
) else (
    echo [WARN] Virtual environment not found, using system Python
)

REM Start backend in new window
echo [3/4] Starting backend server...
start "DeepHash-ASPE Backend" cmd /k "cd /d "%SCRIPT_DIR%backend" && call "%SCRIPT_DIR%.venv\Scripts\activate.bat" && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo [OK] Backend started (port 8000)

REM Check Node.js
echo [4/4] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo [WARN] Node.js not found. Frontend will not start.
    echo Please install Node.js to run the frontend.
    pause
    exit /b 1
)

REM Start frontend in new window
echo [OK] Node.js found
echo Starting frontend...
start "DeepHash-ASPE Frontend" cmd /k "cd /d "%SCRIPT_DIR%frontend" && npm run dev"
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