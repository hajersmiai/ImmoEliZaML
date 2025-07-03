
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt


df = pd.read_csv("cleaned_data_after_imputation .csv")
postal_df = pd.read_csv("code-postaux-belge.csv", delimiter=";")

postal_df.columns = postal_df.columns.str.strip().str.lower().str.replace(" ", "_")
df.columns = df.columns.str.strip().str.lower()


#Thinking of restricting the prediction afetr excluding prices on the highest 1% spectrum 

price_99th = df['price'].quantile(0.99)
print(f"99th percentile of price: {price_99th:.2f}")

df = df[df['price'] <= price_99th]
print(df.shape)


print(df.columns.tolist())
print(postal_df.columns.tolist())


#enrichment with columns based on postcode ['code', 'localite', 'longitude', 'latitude']
postal_df = postal_df.rename(columns={"code": "postcode"})

postal_df = postal_df.drop_duplicates(subset="postcode")
postal_df = postal_df.groupby("postcode").first().reset_index()

columns_to_keep = ['postcode', 'longitude', 'latitude']
postal_df = postal_df[columns_to_keep]

df["postcode"] = df["postcode"].astype(str)
postal_df["postcode"] = postal_df["postcode"].astype(str)

df = df.merge(postal_df, how="left", on="postcode")
df = df.dropna(subset=['longitude', 'latitude']) #drops only one row


print(df.columns.tolist())
print("Dataset shape:", df.shape)


#Now that the database is ready we proceed to change all categorical to numericalcategorical_cols = df.select_dtypes(include='object').columns.tolist()
#Check first:

#First Type to 0/1
df['type'] = df['type'].map({'HOUSE': 0, 'APARTMENT': 1})

#epc and building condition => Ordinal encoding
epc_map = {'missing': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df['epcscore'] = df['epcscore'].map(epc_map)

condition_map = {
    'missing': 0,
    'TO_RESTORE': 1,
    'TO_BE_DONE_UP': 2,
    'TO_RENOVATE': 3,
    'JUST_RENOVATED': 4,
    'GOOD': 5,
    'AS_NEW': 6
}
df['buildingcondition'] = df['buildingcondition'].map(condition_map)

#province 

df = pd.get_dummies(df, columns=['province'], drop_first=True)

from sklearn.preprocessing import OneHotEncoder

top_50 = df['locality'].value_counts().nlargest(50).index
df['locality_grouped'] = df['locality'].apply(lambda x: x if x in top_50 else 'Other')

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
locality_encoded = ohe.fit_transform(df[['locality_grouped']])
locality_feature_names = ohe.get_feature_names_out(['locality_grouped'])
locality_df = pd.DataFrame(locality_encoded, columns=locality_feature_names, index=df.index)
df = pd.concat([df, locality_df], axis=1)

df.drop('locality_grouped', axis=1, inplace=True)
df.drop('locality', axis=1, inplace=True)


#For subtype I had the idea to use label in an order that has to do with price from apartment to castle, but this would cause major data leakage,
# I decided to go safer with the median surface and group them (this is kinda ok for Forest but not for linear regression).
# # Second try I want to use one hot encoder + linear regression? distance based models like linear regression and KNN sensitive to labels/enumaration

ordered_subtypes = df.groupby("subtype")["habitablesurface"].median().sort_values().index
subtype_mapping = {subtype: i for i, subtype in enumerate(ordered_subtypes)}
df["subtype_encoded"] = df["subtype"].map(subtype_mapping)
df.drop(columns=['subtype'], inplace=True) #drop the original column


df.drop(columns=['postcode'], inplace=True)
columns_to_drop = [
    'type', 'subtype', 'province', 'locality',
    'buildingCondition', 'floodZoneType', 'heatingType', 
    'kitchenType', 'gardenOrientation', 'terraceOrientation', 'epcScore'
]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')


df.to_csv("processed_data_for_modeling.csv", index=False)
print("Final cleaned dataset saved as 'processed_data_for_modeling.csv'")

# Drop all remaining object-type columns to avoid XGBoost errors
df = df.drop(columns=df.select_dtypes(include='object').columns, errors='ignore')


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

print("\n✅ Best Params Found:", best_params)
print("✅ Best Validation MAE:", best_score)

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


