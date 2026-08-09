# Wine Quality Prediction

A machine learning web application that predicts wine quality based on its physicochemical properties.

The project uses supervised machine learning to learn the relationship between characteristics such as acidity, alcohol, pH, sulphates, density, and sugar content and the quality score assigned to the wine.

## Live Demo

https://wine-quality-prediction-f1ko.onrender.com

Open the link, fill in the wine properties, and hit Predict. The model will return a predicted quality score between 0 and 10.

## Features

- Predict wine quality from 12 physicochemical features
- Data preprocessing and missing-value handling
- Categorical feature encoding
- Machine learning model training and evaluation
- LightGBM-based regression model
- Saved model and scaler for inference
- Flask web application for real-time predictions
- Simple HTML frontend for entering wine characteristics

## Tech Stack

| Technology       | Purpose                                      |
| ---------------- | -------------------------------------------- |
| Python           | Core programming language                    |
| Pandas           | Data loading and manipulation                |
| NumPy            | Numerical operations                         |
| Scikit-learn     | Preprocessing, data splitting and evaluation |
| LightGBM         | Wine quality regression model                |
| XGBoost          | Model experimentation/comparison             |
| Joblib           | Saving and loading trained model artifacts   |
| Flask            | Backend web application                      |
| HTML/CSS         | Frontend interface                           |
| Jupyter Notebook | Data exploration and experimentation         |
| Git/GitHub       | Version control and project hosting          |

## How It Works

The project has two main stages.

### 1. Model Training

Historical wine-quality data is processed and used to train machine learning models.

```text
Wine Dataset
     ↓
Data Cleaning
     ↓
Missing Value Handling
     ↓
Categorical Encoding
     ↓
Feature Preprocessing
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Save Model + Scaler
```

The target variable is `quality`, while the input features describe the wine's physicochemical properties.

### 2. Prediction

When a user submits wine characteristics through the Flask application:

```text
User Input
    ↓
Flask /predict endpoint
    ↓
Input Validation
    ↓
Feature Preparation
    ↓
Saved Scaler
    ↓
Trained LightGBM Model
    ↓
Predicted Quality Score
    ↓
Result displayed in browser
```

The trained model is loaded when the Flask application starts, so the model does not need to be retrained for every prediction.

## Input Features

The model uses the following features:

- Wine type
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

The target is the wine's quality score. The quality score follows the scale used by the original dataset, where a higher score represents higher perceived wine quality. The model produces a numerical prediction such as `5.76`.

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/aryansalunke/wine-quality-prediction.git
cd wine-quality-prediction
```

### 2. Create a virtual environment

```powershell
py -3.12 -m venv venv
```

### 3. Activate it

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Start the Flask application

```powershell
python app.py
```

Open the application at `http://127.0.0.1:5000`

## Project Structure

```text
wine-quality-prediction/
│
├── app.py                         # Flask web application
├── train_model.py                 # Model training pipeline
├── predict.py                     # Prediction utility
│
├── wine_quality_lgb_model.pkl     # Trained LightGBM model
├── scaler.pkl                     # Saved preprocessing scaler
│
├── winequality-red.csv            # Wine quality dataset
├── model_comparison_results.csv   # Model evaluation results
│
├── templates/
│   └── index.html                 # Web application interface
│
├── wine.ipynb                     # Data exploration
├── wine2.ipynb                    # Experimentation
├── train.ipynb                    # Training experiments
│
├── requirements.txt               # Python dependencies
├── .gitignore                     # Files excluded from Git
└── README.md                      # Project documentation
```

## Model Evaluation

Multiple machine learning approaches were explored during development. The models were evaluated using Root Mean Squared Error (RMSE) on unseen test data. A lower RMSE indicates that the model's predictions are, on average, closer to the actual quality scores.

The current Flask application uses the trained LightGBM regression model for predictions.
