import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# Step: Tuning hyperparameters progressively
params = {
    'objective': 'regression',          #'regression' ➔ regression task.
    'metric': 'rmse',                   # metric: 'rmse' ➔ evaluates the root mean square error.
    'verbosity': -1,                    # verbosity: -1 ➔ removes unnecessary logs.
    'boosting_type': 'gbdt',            # boosting_type: 'gbdt' ➔ classic gradient boosting.
    'learning_rate': 0.02,              # learning_rate: 0.02 ➔ slow and stable learning.
    'num_leaves': 64,                   # num_leaves: 64 ➔ tree complexity.
    'max_depth': 8,                     # max_depth: 8 ➔ maximum tree depth.
    'min_data_in_leaf': 30,             # min_data_in_leaf: 30 ➔ regularization, avoids overfitting.
    'feature_fraction': 0.8,            # feature_fraction: 0.8 ➔ downsamples features at each split.
    'bagging_fraction': 0.8,            # bagging_fraction: 0.8, 
    'bagging_freq': 5,                  # bagging_freq: 5 ➔ Data subsampling for robustness.
    'seed': 42                          # seed: 42 ➔ Reproducibility.


}

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

gbm = lgb.train(
    params,                                         # params ➔ Defined hyperparameters.
    lgb_train,
    num_boost_round=5000,                           # num_boost_round=5000 ➔ Maximum number of iterations (stops before if no improvement).
    valid_sets=[lgb_train, lgb_eval],               # valid_sets ➔ List of training and validation sets.
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),    # early_stopping ➔ Stops if no improvement occurs after 100 iterations.
        lgb.log_evaluation(period=200)              # log_evaluation ➔ Displays logs every 200 rounds.
    ]
)

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)  #Predicts prices on test data with the best number of iterations found.

# Metrics
mae = mean_absolute_error(y_test, y_pred)               # MAE: Mean Absolute Error ➔ Average error in EUR per asset.
rmse = np.sqrt(mean_squared_error(y_test, y_pred))      # RMSE: Root Mean Squared Error ➔ Sensitive to outliers, in EUR.
r2 = r2_score(y_test, y_pred)                           # R²: Proportion of variance explained by the model (0 to 1).

# Displays metrics in a readable way for analysis and progress tracking.
print(f"✅ MAE: {mae:.2f} EUR")
print(f"✅ RMSE: {rmse:.2f} EUR")
print(f"✅ R^2 Score: {r2:.4f}")

# 9️⃣ Plot predicted vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.4, color='royalblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price (€)")
plt.ylabel("Predicted Price (€)")
plt.title("LightGBM: Actual vs Predicted Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

