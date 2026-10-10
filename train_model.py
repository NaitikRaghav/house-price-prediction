
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# 1. Create project directories
# --------------------------------------------------
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 65)
print("          HOUSE PRICE PREDICTION - MODEL TRAINING")
print("=" * 65)

# --------------------------------------------------
# 2. Load the California Housing dataset
# --------------------------------------------------
print("\nLoading California Housing dataset...")

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

# The target MedHouseVal is measured in units of $100,000.
df.rename(columns={"MedHouseVal": "Price"}, inplace=True)

print(f"\nDataset shape: {df.shape}")
print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe().round(2))

# Save a local copy of the dataset
df.to_csv("data/california_housing.csv", index=False)

# --------------------------------------------------
# 3. Explore the dataset
# --------------------------------------------------
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nPrice distribution:")
print(df["Price"].describe())

# Plot the target distribution
plt.figure(figsize=(9, 5))
sns.histplot(df["Price"], bins=40, kde=True)
plt.title("Distribution of California House Values")
plt.xlabel("Median House Value (units of $100,000)")
plt.ylabel("Number of districts")
plt.tight_layout()
plt.savefig("results/price_distribution.png", dpi=150)
plt.close()

# Plot the feature correlation matrix
plt.figure(figsize=(11, 8))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)
plt.title("Housing Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("results/correlation_matrix.png", dpi=150)
plt.close()

# --------------------------------------------------
# 4. Prepare features and target
# --------------------------------------------------
X = df.drop(columns=["Price"])
y = df["Price"]

# Split before fitting preprocessing to prevent leakage
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

# --------------------------------------------------
# 5. Build preprocessing and regression pipeline
# --------------------------------------------------
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

# Train the model
print("\nTraining Linear Regression model...")
pipeline.fit(X_train, y_train)

# --------------------------------------------------
# 6. Evaluate predictions on unseen test data
# --------------------------------------------------
predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\n" + "=" * 65)
print("                    MODEL EVALUATION")
print("=" * 65)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE)  : {mse:.4f}")
print(f"Root Mean Squared Error   : {rmse:.4f}")
print(f"R-squared (R²)            : {r2:.4f}")

print(
    "\nNote: Price values are in units of $100,000. "
    "Multiply MAE and RMSE by $100,000 to express them in dollars."
)

# Save metrics
metrics = pd.DataFrame([{
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2": r2,
    "Training Samples": len(X_train),
    "Testing Samples": len(X_test)
}])
metrics.to_csv("results/evaluation_metrics.csv", index=False)

# Save a sample of actual and predicted values
comparison = pd.DataFrame({
    "Actual Price ($100,000 units)": y_test.to_numpy(),
    "Predicted Price ($100,000 units)": predictions
})
comparison.head(100).to_csv(
    "results/predictions_sample.csv",
    index=False
)

# --------------------------------------------------
# 7. Plot actual vs predicted prices
# --------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.35, s=15)

min_value = min(y_test.min(), predictions.min())
max_value = max(y_test.max(), predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)
plt.title("Actual vs Predicted House Values")
plt.xlabel("Actual Value ($100,000 units)")
plt.ylabel("Predicted Value ($100,000 units)")
plt.tight_layout()
plt.savefig("results/actual_vs_predicted.png", dpi=150)
plt.close()

# --------------------------------------------------
# 8. Save the trained pipeline
# --------------------------------------------------
joblib.dump(pipeline, "models/house_price_model.pkl")

print("\nGenerated files:")
print("- data/california_housing.csv")
print("- models/house_price_model.pkl")
print("- results/price_distribution.png")
print("- results/correlation_matrix.png")
print("- results/actual_vs_predicted.png")
print("- results/evaluation_metrics.csv")
print("- results/predictions_sample.csv")

print("\nModel training completed successfully!")
