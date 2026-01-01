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
python3 -m app.main
