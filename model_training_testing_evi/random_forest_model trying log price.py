
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("cleaned_data_after_imputation .csv")
postal_df = pd.read_csv("code-postaux-belge.csv", delimiter=";")

postal_df.columns = postal_df.columns.str.strip().str.lower().str.replace(" ", "_")
df.columns = df.columns.str.strip().str.lower()

# Thinking of restricting the prediction afetr excluding prices on the highest 1% spectrum 

# price_99th = df['price'].quantile(0.99)
# print(f"95th percentile of price: {price_99th:.2f}")

# df = df[df['price'] <= price_99th]
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

# categorical_cols = df.select_dtypes(include='object').columns.tolist()
# print("Categorical columns:", categorical_cols)

# for col in categorical_cols:
#     print(f"\nColumn: {col}")
#     print(f"Unique values ({df[col].nunique()}):")
#     print(df[col].value_counts())

# print(df.isna().sum())

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
# from sklearn.compose import ColumnTransformer

# ct = ColumnTransformer(transformers=[
#     ('onehot', OneHotEncoder(drop='first'), ['province'])
# ], remainder='passthrough')

# X_transformed = ct.fit_transform(df)

#For localities they are so many is going to bring noice to the model, I checked with bar plot and most listing are in the top 50 localities.(Forest can handle even 100, but linear models no(30 is ok).
#Plus in this occasion it wasnt worth it from 50-100 top localities). I grouped the rest in Other

top_50 = df['locality'].value_counts().nlargest(50).index
df['locality_grouped'] = df['locality'].apply(lambda x: x if x in top_50 else 'Other')

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
locality_encoded = ohe.fit_transform(df[['locality_grouped']])
locality_feature_names = ohe.get_feature_names_out(['locality_grouped'])
locality_df = pd.DataFrame(locality_encoded, columns=locality_feature_names, index=df.index)
df = pd.concat([df, locality_df], axis=1)

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

df.to_csv("data_for_modeling.csv", index=False)
print("Final cleaned dataset saved as 'processed_data_for_modeling.csv'")



target_name = "price"
data_columns = df.columns.drop([target_name, "log_price"], errors='ignore')


X = df[data_columns]
y = df[target_name]

# SPlit to train and a temporary group which is gonna be split in validation adn testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42) #70%

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) # 15%  (calidation group to tune the model and then 15% for testing

print(X_train.describe())

y_train_log = np.log1p(y_train)



# #I evaluated teh most important features and I decided to drop 30 

# # Load the feature importances
# feature_importances = pd.read_csv("feature_importances.csv")

# # Filter only locality_grouped columns
# locality_features = feature_importances[feature_importances['Feature'].str.startswith('locality_grouped_')]

# # Get bottom 20 least important locality features
# low_importance_localities = locality_features.nsmallest(30, 'Importance')['Feature'].tolist()

# # Drop them from your dataset
# X_train = X_train.drop(columns=low_importance_localities, errors='ignore')
# X_val = X_val.drop(columns=low_importance_localities, errors='ignore')
# X_test = X_test.drop(columns=low_importance_localities, errors='ignore')

# print("Dropped these 30 least important locality features:")
# print(low_importance_localities)


#For random forest

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 100, random_state =42)
model.fit(X_train, y_train_log)

log_preds = model.predict(X_val)
y_pred = np.expm1(log_preds)

r2 = r2_score(y_val, y_pred) # Evaluation
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)

print(f"R² score: {r2:.3f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# === SIDE-BY-SIDE COMPARISON OF RAW PRICE vs LOG PRICE MODELS ===

# 1. Train on raw prices
rf_raw = RandomForestRegressor(n_estimators=100, random_state=42)
rf_raw.fit(X_train, y_train)

y_pred_raw = rf_raw.predict(X_val)

r2_raw = r2_score(y_val, y_pred_raw)
mae_raw = mean_absolute_error(y_val, y_pred_raw)
rmse_raw = np.sqrt(mean_squared_error(y_val, y_pred_raw))

# 2. Train on log-transformed prices
y_train_log = np.log1p(y_train)
rf_log = RandomForestRegressor(n_estimators=100, random_state=42)
rf_log.fit(X_train, y_train_log)

log_preds = rf_log.predict(X_val)
y_pred_log = np.expm1(log_preds)

r2_log = r2_score(y_val, y_pred_log)
mae_log = mean_absolute_error(y_val, y_pred_log)
rmse_log = np.sqrt(mean_squared_error(y_val, y_pred_log))

# Print both results
print("\n=== Raw Price Model ===")
print(f"R²: {r2_raw:.3f}, MAE: {mae_raw:.2f}, RMSE: {rmse_raw:.2f}")

print("\n=== Log-Transformed Model ===")
print(f"R²: {r2_log:.3f}, MAE: {mae_log:.2f}, RMSE: {rmse_log:.2f}")


import matplotlib.pyplot as plt

comparison_df = pd.DataFrame({
    'Model': ['Raw Price', 'Log-Transformed'],
    'R²': [r2_raw, r2_log],
    'MAE': [mae_raw, mae_log],
    'RMSE': [rmse_raw, rmse_log]
})

comparison_df.to_csv("model_comparison_log_vs_raw.csv", index=False)
print("Saved comparison results to model_comparison_log_vs_raw.csv")

with open("log_vs_raw_comparison.txt", "w") as f:
    f.write("=== Raw Price Model ===\n")
    f.write(f"R²: {r2_raw:.3f}, MAE: {mae_raw:.2f}, RMSE: {rmse_raw:.2f}\n")
    f.write("\n=== Log-Transformed Model ===\n")
    f.write(f"R²: {r2_log:.3f}, MAE: {mae_log:.2f}, RMSE: {rmse_log:.2f}\n")


plt.scatter(y_val, y_pred, alpha=0.3)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red')
plt.show()

# #Problem with random forest = can overfit without tuning
# #harder to interpret
# #Fine tuning using GridSearchCV


# from sklearn.model_selection import RandomizedSearchCV

# rf = RandomForestRegressor(random_state=42)

# param_grid = {
#     "n_estimators":[200, 300, 400], #n_estimators= number of trees, More trees= higher stability but slower
#     "max_depth":[None, 20, 30, 40], #depth of each tree
#     "max_features":["sqrt", None], #features for each split. sqrt betetr in wide databases
#     "min_samples_split":[2, 4, 6], #minimum samples required to split a node
#     "min_samples_leaf":[1, 2, 4], #minimum samples required at leaf node
#     "bootstrap": [True, False] #whether to use bootstarp samples= bagging= True
# }

# rf_Grid = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, cv=3, verbose=2, random_state=42, n_jobs= -1 )#cv means cross=validation of 3, n_jobs to run the algoryth fast adn verbose should 

# rf_Grid.fit(X_train, y_train)


# rf_Grid.best_params_ #to see the best parameters available
# print("Best Parameters:", rf_Grid.best_params_)

# # Predict using the tuned/best model
# y_val_best = rf_Grid.predict(X_val)

# # Recalculate MAE and RMSE based on tuned model
# mae = mean_absolute_error(y_val, y_val_best)
# rmse = np.sqrt(mean_squared_error(y_val, y_val_best))

# # print them again
# print(f"[TUNED] MAE: {mae:.2f}")
# print(f"[TUNED] RMSE: {rmse:.2f}")


# print(f"Train accuracy = {rf_Grid.score(X_train, y_train):.3f}")
# print(f"Test accuracy = {rf_Grid.score(X_test, y_test):.3f}")
# print(f"Train accuracy = {rf_Grid.score(X_train, y_train):.3f}")
# print(f"Test accuracy = {rf_Grid.score(X_test, y_test):.3f}")

# X_train.to_csv("X_train.csv", index=False)
# X_val.to_csv("X_val.csv", index=False)
# X_test.to_csv("X_test.csv", index=False)
# y_train.to_csv("y_train.csv", index=False)
# y_val.to_csv("y_val.csv", index=False)
# y_test.to_csv("y_test.csv", index=False)


# # Save Evaluation Metrics
# with open("random_forest_initial_metrics.txt", "w") as f:
#     f.write("Random Forest Evaluation (Initial Run)\n")
#     f.write(f"Best Parameters: {rf_Grid.best_params_}\n")
#     f.write(f"Train R² Score: {rf_Grid.score(X_train, y_train):.3f}\n")
#     f.write(f"Test R² Score: {rf_Grid.score(X_test, y_test):.3f}\n")
#     f.write(f"MAE: {mae:.2f}\n")
#     f.write(f"RMSE: {rmse:.2f}\n")

# print("Training/Validation/Test splits saved to CSV.")

# df.to_csv("processed_data_for_modeling.csv", index=False)

# # Predict with best model
# y_val_best = rf_Grid.predict(X_val)

# # Recalculate metrics (optional, but recommended)
# mae = mean_absolute_error(y_val, y_val_best)
# rmse = np.sqrt(mean_squared_error(y_val, y_val_best))
# print(f"[TUNED] MAE: {mae:.2f}")
# print(f"[TUNED] RMSE: {rmse:.2f}")

# # Save updated predictions
# predicted_vs_actual = pd.DataFrame({
#     'Actual Price': y_val,
#     'Predicted Price': y_val_best
# })
# predicted_vs_actual.to_csv("validation_predictions.csv", index=False)

# # Plot again
# plt.figure(figsize=(8, 6))
# plt.scatter(y_val, y_val_best, alpha=0.3)
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title("Validation Set: Actual vs Predicted Prices (Tuned RF)")
# plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red')
# plt.savefig("actual_vs_predicted_scatter_tuned.png")
# plt.close()


# importances = rf_Grid.best_estimator_.feature_importances_
# features = X_train.columns

# feature_importance_df = pd.DataFrame({
#     'Feature': features,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)

# feature_importance_df.to_csv("feature_importances.csv", index=False)
