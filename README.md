# Student Performance Prediction 

A machine learning pipeline for student performance prediction using data ingestion, transformation, and model training.

## Project Overview

This project implements an end-to-end machine learning workflow that:
- Ingests raw student data
- Performs data preprocessing and feature engineering
- Trains multiple regression models
- Evaluates and selects the best performing model

## Project Structure

```
ml_project/
├── artifacts/                 # Generated outputs (train/test splits, preprocessor, model)
├── notebook/                  # Jupyter notebooks for exploration
│   └── data/
│       └── stud.csv          # Raw student dataset
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Load and split data
│   │   ├── data_transformation.py # Preprocessing and feature scaling
│   │   └── model_trainer.py       # Model training and evaluation
│   ├── exception.py           # Custom exception handling
│   ├── logger.py              # Logging configuration
│   └── utils.py               # Utility functions
├── requirements.txt           # Project dependencies
├── setup.py                   # Package installation config
└── README.md                  # This file
```

## Features

### 1. Data Ingestion (`src/components/data_ingestion.py`)
- Reads raw CSV data
- Performs train-test split (80-20)
- Saves splits to `artifacts/` directory

### 2. Data Transformation (`src/components/data_transformation.py`)
- **Numerical Features**: Imputation (median) + Standard Scaling
- **Categorical Features**: Imputation (most frequent) + One-Hot Encoding + Scaling
- Handles columns:
  - **Numerical**: `writing_score`, `reading_score`
  - **Categorical**: `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`
  - **Target**: `math_score`
- Saves preprocessor object for inference

### 3. Model Training (`src/components/model_trainer.py`)
Trains 8 regression models:
- Random Forest
- Decision Tree
- Gradient Boosting
- Linear Regression
- K-Neighbors Regressor
- XGBoost
- CatBoost
- AdaBoost

Selects best model based on R² score on test set.

## Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

1. **Clone/Download the project**
```bash
cd ml_project
```

2. **Create virtual environment**
```bash
python -m venv venv
```

3. **Activate virtual environment**
- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Install project as package** (optional)
```bash
pip install -e .
```

## Usage

### Run the full pipeline

```bash
python src/components/data_ingestion.py
```

This will:
1. Load data from `notebook/data/stud.csv`
2. Split into train/test sets
3. Preprocess and transform data
4. Train all models
5. Save best model to `artifacts/model.pkl`
6. Return R² score

### Output
```
Best model found on both training and testing dataset
R² Score: 0.87 (example)
```

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
catboost
dill
```

See `requirements.txt` for specific versions.

## Project Metadata

- **Name**: ml_project
- **Version**: 0.0.1
- **Author**: Abhisar Dhengare
- **Email**: abhisardhengare@gmail.com

## Error Handling

Custom exception handling with detailed error messages including:
- File name where error occurred
- Line number
- Error message

See `src/exception.py` for implementation.

## Logging

All operations are logged to `logs/` directory with timestamps.

See `src/logger.py` for configuration.

## Artifacts Generated

- `artifacts/train.csv` - Preprocessed training data
- `artifacts/test.csv` - Preprocessed test data
- `artifacts/preprocessor.pkl` - Fitted preprocessor pipeline
- `artifacts/model.pkl` - Best trained model

## Future Enhancements

- Hyperparameter tuning
- Cross-validation
- Feature selection
- Model interpretation (SHAP)
- API deployment
- Prediction pipeline for new data

## License

MIT

## Contact

For questions or issues, contact: abhisardhengare@gmail.com