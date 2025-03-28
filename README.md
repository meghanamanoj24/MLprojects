# Machine Learning Prediction System

This project implements a web-based prediction system using various machine learning models:
- Simple Linear Regression (SLR)
- Multiple Linear Regression (MLR)
- Polynomial Regression
- Logistic Regression
- K-Nearest Neighbors (KNN)

## Features
- Web interface for easy prediction
- Multiple model selection
- Salary prediction using regression models
- Loan approval prediction using classification models
- Modern and responsive UI

## Local Setup Instructions

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Train the models:
```bash
python train_models.py
```

4. Run the Flask application:
```bash
python app.py
```

5. Open your web browser and navigate to:
```
http://localhost:5000
```

## Deployment to Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Python Version: 3.8.12

## Usage

1. Enter the required information:
   - Age
   - Years of Experience
   - Education Level (1-4)
   - Select the desired model

2. Click "Predict" to get the result

## Model Details

- **Simple Linear Regression**: Predicts salary based on experience only
- **Multiple Linear Regression**: Predicts salary based on all features
- **Polynomial Regression**: Predicts salary using polynomial features
- **Logistic Regression**: Predicts loan approval (0 or 1)
- **KNN**: Predicts loan approval using k-nearest neighbors

## Dataset

The system uses a sample dataset (`sample_data.csv`) with the following features:
- Age
- Experience
- Education Level
- Salary
- Loan Approval Status

You can replace the sample dataset with your own data by modifying the `train_models.py` file.

## Project Structure
```
├── app.py                 # Flask application
├── train_models.py        # Model training script
├── sample_data.csv        # Sample dataset
├── requirements.txt       # Python dependencies
├── runtime.txt           # Python version specification
├── Procfile             # Render deployment configuration
├── models/              # Directory for saved models
└── templates/           # HTML templates
    └── index.html       # Main web interface
``` 