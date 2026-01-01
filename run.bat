@echo off
setlocal

echo Checking for virtual environment...
if exist .venv\Scripts\activate.bat (
    echo Activating .venv...
    call .venv\Scripts\activate.bat
) else if exist venv\Scripts\activate.bat (
    echo Activating venv...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Using system Python.
)

echo Starting Multilingual Transcriber...
python -c "import fastapi" >nul 2>&1
if %errorlevel% neq 0 (
    echo Dependencies not found. Installing from requirements.txt...
    pip install -r requirements.txt
)
python -m app.main
pause
endlocal
