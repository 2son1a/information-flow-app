#!/bin/bash
IP_ADDRESS=$1

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

# Run streamlit
echo "Starting Streamlit server..."
echo "Frontend will be available at: http://${IP_ADDRESS}:8501"
streamlit run streamlit_app.py --server.address $IP_ADDRESS --server.port 8501 