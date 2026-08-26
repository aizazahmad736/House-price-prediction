import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# 1. Load dataset
# ==============================

df = pd.read_csv("data/train.csv")

print("Dataset loaded:", df.shape)

# ==============================
# 2. Prepare data
# ==============================

data = df.copy()

# Remove ADDRESS because it is categorical/high-cardinality
data = data.drop("ADDRESS", axis=1)

# Convert categorical columns to numbers
data = pd.get_dummies(data, drop_first=True)

# ==============================
# 3. Separate features and target
# ==============================

X = data.drop("TARGET(PRICE_IN_LACS)", axis=1)
y = data["TARGET(PRICE_IN_LACS)"]

# ==============================
# 4. Train/test split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# ==============================
# 5. Train Random Forest
# ==============================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==============================
# 6. Evaluate
# ==============================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n===== MODEL RESULTS =====")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

# ==============================
# 7. Save model + feature names
# ==============================

model_data = {
    "model": model,
    "features": list(X.columns)
}

joblib.dump(model_data, "house_price_model.pkl")

print("\nModel saved successfully!")
print("File: house_price_model.pkl")