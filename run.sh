#!/bin/bash

# Check for virtual environment
if [ -d ".venv" ]; then
    echo "Activating .venv..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating venv..."
    source venv/bin/activate
else
    echo "No virtual environment found. Using system Python."
fi

echo "Starting Multilingual Transcriber..."
if ! python3 -c "import fastapi" > /dev/null 2>&1; then
    echo "Dependencies not found. Installing from requirements.txt..."
    pip install -r requirements.txt
fi
python3 -m app.main
