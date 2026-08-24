# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the weather sensor data.

2. Extract relevant environmental and time features.

3. Split the data into training and testing sets.
 
4. Train Random Forest models for temperature, PM2.5, and energy.

5. Predict the outputs and calculate MAE, RMSE, and R².
 

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Supriya S J
RegisterNumber: 212224110051

# ============================================================
# RANDOM FOREST REGRESSION
# Prediction of Temperature, PM2.5 and Energy (TSR)
# ============================================================

# Import required libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ============================================================
# 1. LOAD THE DATASET
# ============================================================

df = pd.read_csv(
    "C:\\Users\\admin\\Downloads\\weather-station-eee-block_2024_07_13.csv"
)

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

# Convert time column to datetime format

df["time"] = pd.to_datetime(df["time"])

# Extract date and time features

df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month
df["day"] = df["time"].dt.day
df["hour"] = df["time"].dt.hour
df["minute"] = df["time"].dt.minute


# Select required columns

features = [
    "hum",
    "pressure",
    "wind_speed",
    "year",
    "month",
    "day",
    "hour",
    "minute"
]

targets = [
    "tem",      # Temperature
    "pm2_5",    # PM2.5 Pollution Level
    "tsr"       # Energy / Solar Radiation
]


# Create a new dataset containing selected features and targets

data = df[features + targets]

# Remove missing values

data = data.dropna()

print("\nDataset shape after removing missing values:")
print(data.shape)


# ============================================================
# 3. DEFINE INPUT AND OUTPUT VARIABLES
# ============================================================

X = data[features]

y_temperature = data["tem"]
y_pm25 = data["pm2_5"]
y_energy = data["tsr"]


# ============================================================
# 4. SPLIT THE DATA
# ============================================================

X_train, X_test, y_temp_train, y_temp_test = train_test_split(
    X,
    y_temperature,
    test_size=0.2,
    random_state=42
)

# PM2.5 train-test split

X_train_pm, X_test_pm, y_pm_train, y_pm_test = train_test_split(
    X,
    y_pm25,
    test_size=0.2,
    random_state=42
)

# Energy train-test split

X_train_en, X_test_en, y_energy_train, y_energy_test = train_test_split(
    X,
    y_energy,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 5. CREATE RANDOM FOREST MODELS
# ============================================================

temperature_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

pm25_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

energy_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# ============================================================
# 6. TRAIN THE MODELS
# ============================================================

temperature_model.fit(X_train, y_temp_train)

pm25_model.fit(X_train_pm, y_pm_train)

energy_model.fit(X_train_en, y_energy_train)


# ============================================================
# 7. MAKE PREDICTIONS
# ============================================================

temp_predictions = temperature_model.predict(X_test)

pm25_predictions = pm25_model.predict(X_test_pm)

energy_predictions = energy_model.predict(X_test_en)


# ============================================================
# 8. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(actual, predicted, model_name):

    r2 = r2_score(actual, predicted)

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    print("\n==========================================")
    print(model_name)
    print("==========================================")

    print("R2 Score :", round(r2, 4))
    print("MAE      :", round(mae, 4))
    print("RMSE     :", round(rmse, 4))


# ============================================================
# 9. EVALUATE TEMPERATURE MODEL
# ============================================================

evaluate_model(
    y_temp_test,
    temp_predictions,
    "TEMPERATURE PREDICTION MODEL"
)


# ============================================================
# 10. EVALUATE PM2.5 MODEL
# ============================================================

evaluate_model(
    y_pm_test,
    pm25_predictions,
    "PM2.5 PREDICTION MODEL"
)


# ============================================================
# 11. EVALUATE ENERGY MODEL
# ============================================================

evaluate_model(
    y_energy_test,
    energy_predictions,
    "ENERGY / TSR PREDICTION MODEL"
)


# ============================================================
# 12. CROSS-VALIDATION
# ============================================================

print("\n==========================================")
print("CROSS-VALIDATION RESULTS")
print("==========================================")


# Temperature Cross Validation

temp_cv = cross_val_score(
    temperature_model,
    X,
    y_temperature,
    cv=5,
    scoring="r2"
)

print("\nTemperature Cross Validation R2 Scores:")
print(temp_cv)

print(
    "Mean CV R2 Score:",
    round(temp_cv.mean(), 4)
)


# PM2.5 Cross Validation

pm25_cv = cross_val_score(
    pm25_model,
    X,
    y_pm25,
    cv=5,
    scoring="r2"
)

print("\nPM2.5 Cross Validation R2 Scores:")
print(pm25_cv)

print(
    "Mean CV R2 Score:",
    round(pm25_cv.mean(), 4)
)


# Energy Cross Validation

energy_cv = cross_val_score(
    energy_model,
    X,
    y_energy,
    cv=5,
    scoring="r2"
)

print("\nEnergy Cross Validation R2 Scores:")
print(energy_cv)

print(
    "Mean CV R2 Score:",
    round(energy_cv.mean(), 4)
)


# ============================================================
# 13. DISPLAY SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({

    "Actual Temperature": y_temp_test.values,
    "Predicted Temperature": temp_predictions

})

print("\nSample Temperature Predictions:")
print(results.head(10))


# PM2.5 predictions

pm_results = pd.DataFrame({

    "Actual PM2.5": y_pm_test.values,
    "Predicted PM2.5": pm25_predictions

})

print("\nSample PM2.5 Predictions:")
print(pm_results.head(10))


# Energy predictions

energy_results = pd.DataFrame({

    "Actual Energy (TSR)": y_energy_test.values,
    "Predicted Energy (TSR)": energy_predictions

})

print("\nSample Energy Predictions:")
print(energy_results.head(10))
*/
```

## Output:

<img width="846" height="465" alt="image" src="https://github.com/user-attachments/assets/19aeca22-e1a5-4630-97b1-ee2d14e3b658" />

<img width="601" height="509" alt="image" src="https://github.com/user-attachments/assets/f8b55c3c-c0fc-4cd6-9475-0ba47d564892" />

<img width="506" height="440" alt="image" src="https://github.com/user-attachments/assets/5bbbcf54-2d28-4937-8920-39ae80300503" />

<img width="859" height="818" alt="image" src="https://github.com/user-attachments/assets/56fd089c-5df8-4449-ba67-7aa01a3f524a" />

<img width="668" height="819" alt="image" src="https://github.com/user-attachments/assets/457d3279-2be0-4157-a438-0ad729543f95" />


## Result:
Thus the program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm is Completed.
