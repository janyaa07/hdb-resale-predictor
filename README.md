# HDB Resale Price Predictor

A Streamlit app for predicting HDB resale prices using a trained neural network model and additional features (CPI, amenities).

---

##  Quick Start

### 1. Clone this repository

```bash
git clone https://github.com/janyaa07/hdb-resale-predictor.git
cd hdb-resale-predictor
```

### 2. Create a virtual environment

####  Windows (PowerShell)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

####  macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## Project Structure

This repository contains all necessary files for building, training, evaluating, and deploying an HDB resale price prediction model with a Streamlit UI.

---

### Jupyter Notebooks

| File Name | Description |
|-----------|-------------|
| `Initial_Setup.ipynb` | Prepares the data pipeline, performs initial cleaning, and merges datasets including CPI and geolocation info. |
| `Final_Model_Neural_Network.ipynb` | Trains and evaluates the final neural network model using PyTorch, including hyperparameter tuning and performance visualization. |
| `Random_Forest_Regressor_SVR.ipynb` | Compares Random Forest and Support Vector Regressor models using the same dataset and evaluation metrics. |
| `XG_Boost_KNN.ipynb` | Explores the performance of XGBoost and K-Nearest Neighbors (with GridSearch tuning). |


---

### CSV Files

| File Name | Description |
|-----------|-------------|
| `df_merged.csv` | Final processed dataset with all training features (e.g., CPI, lease, location, amenities). |
| `address_data_full.csv` | Contains lat/lon and distances to nearest MRT, malls, and police centers for autofill functionality. |
| `M213751 (1).csv` | Contains historical CPI index values used for calculating inflation-adjusted prices. |
| `prediction_log.csv` | Auto-generated file that logs all predictions made via the Streamlit app. |
| `ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv` | Base Dataset containing HDB prices information from 2017 to 2025. |

---

### Model & Supporting Files

| File Name | Description |
|-----------|-------------|
| `app.py` | Main Streamlit application file for the HDB Resale Price Predictor UI. |
| `hdb_model.pt` | Trained PyTorch model used for live prediction. |
| `scaler.pkl` | Serialized Scikit-learn StandardScaler object used to normalize inputs before prediction. |

---

###  Others

| File Name | Description |
|-----------|-------------|
| `requirements.txt` | List of all Python packages required to run the project. |
| `.gitignore` | Specifies files and folders to be ignored by Git (e.g., logs, cache, venv). |

---
