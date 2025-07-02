# 🏠 ImmoEliza ML — Belgium Real Estate Price Prediction

---

## 📌 Project Overview

**ImmoEliza ML** is a complete **machine learning pipeline** designed to predict **real estate sales prices in Belgium** accurately.

This project includes:
- Scraping and cleaning Belgian real estate data.
- Preparing data for supervised machine learning.
- Training and evaluating regression models:
  - Linear Regression
  - LightGBM
  - CatBoost
  - XGBoost
  - Stacking models
- Generating clear metrics (MAE, RMSE, R²) for business reporting.
- A scalable structure ready for further API or batch deployment.

---

## 🚀 Features

✅ End-to-end ML pipeline (data → model → evaluation)  
✅ Outlier removal, missing value handling, and encoding ready  
✅ Model selection and comparative evaluation  
✅ Fast iteration and hyperparameter tuning-ready  
✅ Clear, well-structured, typed, and documented Python code  
✅ Black formatted for clean style

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-org/immoeliza-ml.git
cd immoeliza-ml
```
### 2️⃣ Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
## 🛠️ Usage

### Running the pipeline

Execute one of the provided scripts:
```bash
python src/linear_regression.py
python src/lightGBM.py
python src/CatBoost.py
python src/XGBoost.py
python src/stacking.py
```
## ✅ You will get:

    MAE, RMSE, and R² metrics printed.

    Visualizations comparing y_test vs y_pred for clear validation.

    Early stopping logs for monitoring training.

## 📊 Results

Using XGBoost with tuned hyperparameters on 70,211 Belgian real estate records with 26 engineered features:

✅ XGBoost MAE: 67537.51 EUR

✅ XGBoost RMSE: 94737.04 EUR

✅ XGBoost R^2: 0.6579

🔹 These results demonstrate strong predictive capacity for property pricing using ML.
🔹 We can further improve these results through:

    Advanced feature engineering (ratios, location clusters)

    Adding external socioeconomic and geospatial data

    Automated hyperparameter tuning (Optuna, Bayesian optimization)

to enhance predictive accuracy for ImmoEliza’s deployment in the Belgian market.

## 🖼️ Visuals

    Correlation heatmaps for feature selection.

    Scatter plots of actual vs predicted prices.

    Validation curves for early stopping.

## 📈 Project Structure

immoeliza-ml/
│
├── data/                  # Raw and cleaned CSV datasets
├── src/                   # Pipeline scripts, data cleaning modules, utils
├── outputs/               # Scatter plots of actual via predicted prices for each regression model
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

## 💡 Improvement Ideas

✅ Integrate socioeconomic and neighborhood indicators.
✅ Automate hyperparameter tuning with Optuna or GridSearchCV.
✅ Deploy as a FastAPI endpoint for real-time prediction.
✅ Integrate SHAP for interpretability on pricing predictions.
📝 License

MIT License. Feel free to use, adapt, and contribute.
🤝 Contributions

Contributions are welcome! Please open an issue or submit a pull request for improvements, bug fixes, or documentation enhancements.
