.PHONY: setup install download-data eda train predict clean

# Setup environment
setup:
	@echo "Setting up environment..."
	@chmod +x setup.sh
	@./setup.sh

# Install dependencies
install:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt

# Download competition data
download-data:
	@echo "Downloading competition data..."
	@cd data/raw && kaggle competitions download -c playground-series-s5e7
	@cd data/raw && unzip -o playground-series-s5e7.zip
	@cd data/raw && rm -f playground-series-s5e7.zip

# Run exploratory data analysis
eda:
	@echo "Running EDA..."
	@cd notebooks && python 01_exploratory_data_analysis.py

# Train model
train:
	@echo "Training model..."
	@cd notebooks && python 02_model_training.py
# Make predictions
predict:
	@echo "Making predictions..."
	@python src/predict.py

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -rf models/*.pkl
	@rm -rf submissions/*.csv
	@rm -rf data/processed/*
	@find . -name "__pycache__" -type d -exec rm -rf {} +
	@find . -name "*.pyc" -delete

# Run full pipeline
all: setup download-data eda train
	@echo "Full pipeline completed!"