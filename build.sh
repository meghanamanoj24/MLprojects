#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Starting build process..."

# Create models directory
echo "Creating models directory..."
mkdir -p models

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Train the models
echo "Training models..."
python train_models.py

echo "Build process completed!" 