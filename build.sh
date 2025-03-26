#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Create models directory
mkdir -p models

# Train the models
python train_models.py 