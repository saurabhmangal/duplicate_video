@echo off
echo ╔══════════════════════════════════════╗
echo ║   Duplicate Video Detector - Setup   ║
echo ╚══════════════════════════════════════╝
echo.

cd /d "%~dp0backend"

echo [1/3] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo [2/3] Installing dependencies...
pip install -r requirements.txt --quiet

echo [3/3] Starting server at http://localhost:8000
echo.
echo  Open your browser at: http://localhost:8000
echo  Press Ctrl+C to stop.
echo.
python main.py
