import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from catboost import CatBoostRegressor, Pool
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error, mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
from scipy import sparse
import scipy.stats as stats


# =========================
# 1. Load and Clean Dataset
# =========================
df = pd.read_csv("data/cleaned_data_after_imputation.csv")

# --- Remove outliers from price using IQR ---
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df["price"] >= lower_bound) & (df["price"] <= upper_bound)]

# ===============================
# 2. Define Features and Targets
# ===============================
numeric_columns = [
    "bedroomCount", "toilet_and_bath", "habitableSurface",
    "facedeCount", "hasTerrace", "totalParkingCount"
]

categorical_columns = [
    "type", "subtype", "province", "locality",
    "postCode", "buildingCondition", "epcScore"
]

for col in categorical_columns:
    df[col] = df[col].astype(str).fillna("nan")

X = df[numeric_columns + categorical_columns]
y = df["price"]

# ==================================
# 3. Split into Train / Val / Test
# ==================================

# First, train vs temp (70/30)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Then, split temp into val/test (50/50 of 30% → 15% each)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Train size: {X_train.shape[0]}")
print(f"Validation size: {X_val.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

# ============================
# 4. Train CatBoost Model
# ============================
catboost_model = CatBoostRegressor(
    iterations=900,
    learning_rate=0.23,
    depth=7,
    loss_function='RMSE',
    random_state=42,
    verbose=100
)

catboost_model.fit(
    X_train, y_train,
    cat_features=categorical_columns,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50
)

# ============================
# 5. Evaluate Performance
# ============================

def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Set Metrics:")
    print("R²:", round(r2_score(y_true, y_pred), 3))
    print("MAE:", round(mean_absolute_error(y_true, y_pred), 2))
    print("MAPE:", round(mean_absolute_percentage_error(y_true, y_pred), 2))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_true, y_pred)), 2))

# Predictions
evaluate_model("Train", y_train, catboost_model.predict(X_train))
evaluate_model("Validation", y_val, catboost_model.predict(X_val))
evaluate_model("Test", y_test, catboost_model.predict(X_test))

# ============================
# 6. Save Model
# ============================
catboost_model.save_model("catboost_model.cbm")
print("\n Model trained and saved as 'catboost_model.cbm'")
