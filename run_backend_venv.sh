#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "backend_venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv backend_venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source backend_venv/bin/activate

cd backend

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Get IP address for macOS
IP_ADDRESS=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1)

# Run uvicorn
echo "Starting uvicorn server..."
echo "Backend will be available at: http://${IP_ADDRESS}:8000"
uvicorn attention.api:app --reload --host 0.0.0.0 --port 8000 