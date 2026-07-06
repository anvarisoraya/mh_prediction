import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url)
data = df['Passengers'].values.astype(np.float32)

plt.figure(figsize=(10, 4))
plt.plot(data, color='steelblue', linewidth=2)
plt.title('International Airline Passengers (1949-1960)')
plt.xlabel('Months')
plt.ylabel('Count')
plt.show()

train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
test_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()

def create_dataset(series, window_size=12):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

WINDOW_SIZE = 12
X_train, y_train = create_dataset(train_scaled, WINDOW_SIZE)
X_test, y_test = create_dataset(test_scaled, WINDOW_SIZE)

X_tr = torch.tensor(X_train).unsqueeze(-1).float()
y_tr = torch.tensor(y_train).unsqueeze(-1).float()
X_te = torch.tensor(X_test).unsqueeze(-1).float()
y_te = torch.tensor(y_test).unsqueeze(-1).float()



class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,     
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)       
        last_step = lstm_out[:, -1, :]   
        return self.fc(last_step)

X_tr = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_te = torch.tensor(X_test,  dtype=torch.float32).unsqueeze(-1)
y_te = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(-1)

print(f'LSTM input shape: {X_tr.shape}  (batch, timesteps, features)')

lstm_model = LSTMForecaster()
print('\nLSTM architecture:')
print(lstm_model)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

EPOCHS = 800
train_losses = []

for epoch in range(EPOCHS):
    lstm_model.train()
    pred = lstm_model(X_tr)
    loss = criterion(pred.squeeze(-1), y_tr.squeeze(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch+1:3d}/{EPOCHS}  Loss: {loss.item():.5f}')

print('Training complete!')




lstm_model.eval()
with torch.no_grad():
    lstm_preds = lstm_model(X_te).numpy().flatten()

lstm_preds_actual = scaler.inverse_transform(lstm_preds.reshape(-1, 1)).flatten()


y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# 4. Calculate metrics using the matched lengths
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae_lstm  = mean_absolute_error(y_test_actual, lstm_preds_actual)
rmse_lstm = np.sqrt(mean_squared_error(y_test_actual, lstm_preds_actual))

print(f'LSTM Performance — MAE: {mae_lstm:.3f}  RMSE: {rmse_lstm:.3f}')



plt.figure(figsize=(10, 4))
plt.plot(y_test_actual, label='Actual Passengers', color='steelblue', marker='o')
plt.plot(lstm_preds_actual, label='LSTM Predicted', color='darkorange', linestyle='--', marker='s')
plt.title('Air Passengers Forecast Evaluation')
plt.xlabel('Months in Test Set')
plt.ylabel('Passengers Count')
plt.legend()
plt.show()