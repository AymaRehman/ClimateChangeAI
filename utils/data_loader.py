import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(path="data/raw_data.csv", lags=5):
    """
    Load GlobalTemperatures dataset and create lag features for time-series prediction.
    """
    # Load dataset
    data = pd.read_csv(path)
    
    # Convert date to datetime and extract year
    data['dt'] = pd.to_datetime(data['dt'])
    data['Year'] = data['dt'].dt.year

    # Use LandAverageTemperature as main target variable
    data = data[['Year', 'LandAverageTemperature']].dropna()
    data.rename(columns={'LandAverageTemperature': 'Temp_Anomaly'}, inplace=True)
    
    # Create lag features for previous years
    for i in range(1, lags + 1):
        data[f'Lag_{i}'] = data['Temp_Anomaly'].shift(i)
    
    # Drop rows with NaN (due to lag creation)
    data.dropna(inplace=True)

    # Split features and target
    X = data[[f'Lag_{i}' for i in range(1, lags + 1)]].values
    y = data['Temp_Anomaly'].values

    # Standardize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Split into train, validation, test
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y
