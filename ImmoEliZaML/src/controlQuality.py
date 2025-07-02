import pandas as pd
import numpy as np

# === CONFIGURATION ===
FILE_PATH = "ImmoEliZaML/data/cleaned_data_no_outliers.csv"  # adapter selon ton projet

# === CHARGEMENT DU FICHIER ===
df = pd.read_csv(FILE_PATH)
print(f"✅ Fichier chargé : {FILE_PATH} ({df.shape[0]} lignes, {df.shape[1]} colonnes)")

# === 1️⃣ POURCENTAGE DE NAN PAR COLONNE ===
nan_percent = df.isnull().mean() * 100

# === 2️⃣ POURCENTAGE DE CHAMPS VIDES PAR COLONNE ('' ou ' ') ===
empty_percent = (df.apply(lambda x: x.astype(str).str.strip() == '').sum() / len(df)) * 100

# === 3️⃣ TABLEAU SYNTHÉTIQUE QUALITÉ DATA ===
quality_df = pd.DataFrame({
    "DataType": df.dtypes,
    "NonNullCount": df.notnull().sum(),
    "NullCount": df.isnull().sum(),
    "NaN%": nan_percent.round(2),
    "Empty%": empty_percent.round(2),
    "UniqueValues": df.nunique()
})

# Tri selon le % de NaN décroissant pour prioriser le nettoyage
quality_df = quality_df.sort_values(by="NaN%", ascending=False)

print("\n📊 Rapport qualité des données :")
print(quality_df)

# === 4️⃣ STATS RAPIDES df.describe() POUR CONTEXTUALISER ===
print("\n📈 Aperçu des statistiques descriptives :")
print(df.describe().T)

# === 5️⃣ EXPORT FACULTATIF POUR SUIVI ===
quality_df.to_csv("data_quality_report.csv")
print("\n✅ Rapport qualité exporté : data_quality_report.csv")
