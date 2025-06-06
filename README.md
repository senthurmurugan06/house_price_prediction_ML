# House Price Prediction

This project implements a machine learning model to predict house prices using the Kaggle House Prices: Advanced Regression Techniques dataset.

## Problem Statement
The goal is to predict the final sale price of homes based on various features such as square footage, number of bedrooms, location, and other characteristics.

## Dataset Description
The dataset contains 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa. The dataset is split into:
- Training set (train.csv): Contains the target variable (SalePrice)
- Test set (test.csv): Contains the same features but without the target variable

## Tools Used
- Python 3.x
- Pandas: Data manipulation and analysis
- NumPy: Numerical computing
- Scikit-learn: Machine learning algorithms
- Matplotlib & Seaborn: Data visualization
- Flask: Web application deployment

## Project Structure
```
house_price_prediction/
│
├── data/
│   └── train.csv, test.csv
├── notebooks/
│   └── house_price_prediction.ipynb
├── models/
│   └── model.pkl
├── main.py
├── requirements.txt
└── README.md
```

## Evaluation Metrics
The models are evaluated using:
- R² Score: Measures the proportion of variance in the dependent variable that's predictable from the independent variables
- RMSE (Root Mean Square Error): Measures the average magnitude of the prediction errors

## Setup and Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from Kaggle and place it in the `data` directory
4. Run the Jupyter notebook or main.py

## Usage
1. For development and exploration, use the Jupyter notebook in the `notebooks` directory
2. For production, run the Flask application:
   ```bash
   python main.py
   ```

## Results
The project implements both Linear Regression and Decision Tree models, comparing their performance in predicting house prices. The results include:
- Model performance metrics (R² and RMSE)
- Visualization of actual vs. predicted prices
- Feature importance analysis

## Future Improvements
- Implement more advanced models (Random Forest, XGBoost)
- Add feature engineering techniques
- Implement cross-validation
- Add model interpretability analysis 