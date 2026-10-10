# House Price Prediction Using Machine Learning

A Machine Learning project that predicts California district-level median house values using Linear Regression. The project includes data preprocessing, model evaluation, visualizations, and an interactive Streamlit web application.

## Features

- California Housing dataset with 20,640 records
- Data exploration and preprocessing
- Linear Regression model
- Feature scaling using StandardScaler
- Evaluation using MAE, MSE, RMSE, and R²
- Actual vs. predicted visualization
- Correlation matrix and price distribution plots
- Interactive Streamlit application

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib
- Streamlit

## Project Structure

```text
house_price_prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
├── models/
└── results/
```

## Installation and Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/NaitikRaghav/house-price-prediction.git
   cd house-price-prediction
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Train the model if the saved model is not available:

   ```bash
   python train_model.py
   ```

5. Start the web application:

   ```bash
   streamlit run app.py
   ```

## Model Evaluation

The Linear Regression model was evaluated using an 80/20 train-test split.

| Metric | Result |
|---|---:|
| MAE | 0.5332 |
| MSE | 0.5559 |
| RMSE | 0.7456 |
| R² Score | 0.5758 |

The target values are expressed in units of $100,000. These results are from the current training run and may vary if the dataset or training configuration changes.

## Important Note

This project uses the California Housing dataset. It predicts median house values at the district level, not the exact market price of an individual property. The app's rupee conversion is a currency-display conversion and does not make this a model trained on Indian property data.

## Author

Naitik Raghav

GitHub: https://github.com/NaitikRaghav

## Purpose

Created as an academic Machine Learning project for learning regression, data preprocessing, model evaluation, and deployment using Streamlit.
