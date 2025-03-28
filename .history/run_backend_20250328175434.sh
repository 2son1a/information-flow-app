#!/bin/bash

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "backend"; then
    echo "Creating conda environment 'backend'..."
    conda create -n backend python=3.11 -y
fi

# Activate conda environment
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate backend


# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Run uvicorn
echo "Starting uvicorn server..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000
