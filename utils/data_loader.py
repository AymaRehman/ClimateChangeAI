import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(path='data/raw_data.csv', lags=5):
    """
    Load Berkeley Earth dataset and create lag features for time-series prediction.
    """
    # Load dataset
    data = pd.read_csv(path)
    data = data[['Year', 'Mean']].dropna()
    data.rename(columns={'Mean':'Temp_Anomaly'}, inplace=True)
    
    # Create lag features
    for i in range(1, lags+1):
        data[f'lag_{i}'] = data['Temp_Anomaly'].shift(i)
    data = data.dropna()
    
    # Features and target
    X = data[[f'lag_{i}' for i in range(1, lags+1)]].values
    y = data['Temp_Anomaly'].values
    
    # Train/validation/test split (chronological)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    
    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1,1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1,1))
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, scaler_X, scaler_y
