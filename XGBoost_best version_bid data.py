
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


df = pd.read_csv("cleaned_data_no_outliers.csv")
postal_df = pd.read_csv("code-postaux-belge.csv", delimiter=";")

postal_df.columns = postal_df.columns.str.strip().str.lower().str.replace(" ", "_")
df.columns = df.columns.str.strip().str.lower()


# Drop raw categorical columns that have been encoded
columns_to_drop = [
    'postcode', 'type', 'subtype', 'province', 'locality',
    'buildingCondition', 'floodZoneType', 'heatingType', 
    'kitchenType', 'gardenOrientation', 'terraceOrientation', 'epcScore'
]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Drop all remaining object-type columns (to avoid issues in modeling)
df = df.drop(columns=df.select_dtypes(include='object').columns, errors='ignore')

# Save the clean, fully numerical dataset
df.to_csv("processed_data_for_modeling.csv", index=False)
print("✅ Final cleaned dataset saved as 'processed_data_for_modeling.csv'")

print([col for col in df.columns if 'subtype' in col])

print(df.shape[1])  # total columns


target_name = "price"
data_columns = df.columns.drop([target_name], errors='ignore')


X = df[data_columns]
y = df[target_name]

# SPlit to train and a temporary group which is gonna be split in validation adn testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42) #70%

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) # 15%  (calidation group to tune the model and then 15% for testing

print(X_train.describe())



from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

# ---------------------
# Evaluation function
# ---------------------
def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(y_val)
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"\n{name} Results:")
    print(f"R² Score: {r2:.3f}")
    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    return model, r2, mae, rmse

# ---------------------

# ---------------------
# Base model
# ---------------------

# ---------------------

# ---------------------
import xgboost as xgb

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define parameter sets to try manually
param_grid = [
    {'learning_rate': 0.03, 'max_depth': 4, 'reg_alpha': 0.5, 'reg_lambda': 1.0},
    {'learning_rate': 0.01, 'max_depth': 5, 'reg_alpha': 0.1, 'reg_lambda': 0.5},
    {'learning_rate': 0.05, 'max_depth': 6, 'reg_alpha': 0.3, 'reg_lambda': 2.0},
]

best_model = None
best_score = float('inf')
best_params = None

for params in param_grid:
    full_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'seed': 42,
        **params
    }

    model = xgb.train(
        full_params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'validation')],
        early_stopping_rounds=20,
        verbose_eval=False
    )

    preds = model.predict(dval)
    mae = mean_absolute_error(y_val, preds)
    print(f"Params: {params} → Validation MAE: {mae:,.2f}")

    if mae < best_score:
        best_score = mae
        best_model = model
        best_params = params

print("\n Best Params Found:", best_params)
print(" Best Validation MAE:", best_score)

# Save model
best_model.save_model("xgb_model_final.json")

# Predict on test set
preds_test = best_model.predict(dtest)
r2 = r2_score(y_test, preds_test)
mae = mean_absolute_error(y_test, preds_test)
rmse = np.sqrt(mean_squared_error(y_test, preds_test))

# Evaluate on training set (optional)
preds_train = best_model.predict(dtrain)
train_mae = mean_absolute_error(y_train, preds_train)
train_r2 = r2_score(y_train, preds_train)

print("\nTraining Set Results:")
print(f"R² Score: {train_r2:.3f}")
print(f"MAE: {train_mae:,.2f}")


print("\nTest Set Results:")
print(f"R² Score: {r2:.3f}")
print(f"MAE: {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")

# Optional: Plot predictions
plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds_test, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Final XGBoost Predictions vs Actual")
plt.grid()
plt.tight_layout()
plt.show()

# Optional: Save predictions
pd.DataFrame({
    'Actual': y_test,
    'Predicted': preds_test
}).to_csv("xgb_predictions_vs_actual.csv", index=False)
