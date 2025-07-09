#!/bin/bash

# Setup script for Kaggle competition
echo "Setting up Kaggle Introvert vs Extrovert competition environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/processed models submissions notebooks/outputs

# Download competition data (requires Kaggle API setup)
echo "Downloading competition data..."
if command -v kaggle &> /dev/null; then
    cd data/raw
    kaggle competitions download -c playground-series-s5e7
    unzip -o playground-series-s5e7.zip
    rm playground-series-s5e7.zip
    cd ../..
    echo "Data downloaded successfully!"
else
    echo "Kaggle CLI not found. Please install and configure Kaggle API."
    echo "Run: pip install kaggle"
    echo "Then setup your API credentials."
fi

echo "Setup complete!"
echo "To get started:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run EDA: python notebooks/01_exploratory_data_analysis.py"
echo "3. Train model: python notebooks/02_model_training.py"