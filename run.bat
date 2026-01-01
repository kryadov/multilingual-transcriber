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
python -m app.main
pause
endlocal
