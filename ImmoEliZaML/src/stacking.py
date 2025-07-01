#LightGBM + CatBoost + LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("ImmoEliZaML/data/cleaned_data_no_outliers.csv")

features = [
    "bedroomCount", "bathroomCount", "habitableSurface", "roomCount",
    "buildingConstructionYear", "facedeCount", "floorCount", "toiletCount",
    "diningRoomSurface", "kitchenSurface", "terraceSurface", "livingRoomSurface",
    "landSurface", "gardenSurface", "parkingCountIndoor", "parkingCountOutdoor",
    "buildingCondition_enc", "epcScore_enc", "heatingType_enc", "floodZoneType_enc", "kitchenType_enc"
]
target = "price"
features = [col for col in features if col in df.columns]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Base models
lgb_model = LGBMRegressor(
    learning_rate=0.02,
    num_leaves=64,
    max_depth=8,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    force_row_wise=True
)

cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.02,
    depth=8,
    verbose=0,
    random_seed=42
)

final_estimator= Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])
# Stacking Regressor
stack = StackingRegressor(
    estimators=[
        ('lgb', lgb_model),
        ('cat', cat_model)
    ],
    final_estimator= final_estimator,
    passthrough=True,
    n_jobs=-1
)

# Fit
stack.fit(X_train, y_train)

# Predict
y_pred = stack.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Stacking MAE: {mae:.2f} EUR")
print(f"✅ Stacking RMSE: {rmse:.2f} EUR")
print(f"✅ Stacking R^2: {r2:.4f}")
 
 #  Plot predicted vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.4, color='royalblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price (€)")
plt.ylabel("Predicted Price (€)")
plt.title("Stacking: Actual vs Predicted Prices")
plt.grid(True)
plt.tight_layout()
plt.show()