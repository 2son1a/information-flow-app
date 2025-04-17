#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "frontend_venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv frontend_venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source frontend_venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Get IP address for macOS
IP_ADDRESS="152.67.255.214"

# Run streamlit
echo "Starting Streamlit server..."
echo "Frontend will be available at: http://${IP_ADDRESS}:8501"
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501 