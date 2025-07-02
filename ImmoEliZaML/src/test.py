import pandas as pd

df = pd.read_csv("ImmoEliZaML/data/cleaned_data_no_outliers.csv")
print(df['gardenOrientation'].array.unique())