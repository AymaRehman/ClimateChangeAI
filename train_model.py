"""
train_model.py
Task 2d: Model Training
----------------------------------------
Trains a simple feedforward neural network to predict temperature anomalies
using lagged historical data.
"""

import torch
from torch import nn, optim
from utils.data_loader import load_and_preprocess
from model.climate_model import TempPredictor, save_model

# --- 1. Load and preprocess data ---
print("Loading and preprocessing data...")
X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = load_and_preprocess(
    path='data/raw_data.csv',
    lags=5
)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# --- 2. Initialize model, loss, optimizer ---
input_size = X_train.shape[1]
model = TempPredictor(input_size=input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 3. Train loop ---
epochs = 200
print("Starting training...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")

# --- 4. Save trained model ---
save_model(model, path='model/trained_model.pth')
print("✅ Training complete. Model saved to 'model/trained_model.pth'")
