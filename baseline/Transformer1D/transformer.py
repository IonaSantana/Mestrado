import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import torch.nn as nn

# 1. Carregar os dados do arquivo .npz
def load_npz_data(file_path):
    data = np.load(file_path)
    return data['data'][:, :, 0]  # Selecionar apenas o atributo de fluxo

# 2. Dividir os dados em treino, validação e teste
def split_data(data, train_ratio=0.6, val_ratio=0.2):
    n = data.shape[0]
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return data[:train_end, :], data[train_end:val_end, :], data[val_end:, :]

# 3. Definir Transformer 1D
class Transformer1D(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(Transformer1D, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.transformer(x)
        return self.fc(x).squeeze(-1)

# 4. Avaliar o modelo
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, rmse, mae

def mape(y_true, y_pred, epsilon=1e-5):
    mask = np.abs(y_true) > epsilon
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

# 5. Pipeline para todos os sensores usando Transformer 1D
def pipeline_all_sensors(file_path, results_file, train_ratio=0.6, val_ratio=0.2):
    data = load_npz_data(file_path)
    num_sensors = data.shape[1]
    train, val, test = split_data(data, train_ratio, val_ratio)
    results = []

    model = Transformer1D(input_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for sensor_id in range(num_sensors):
        train_sensor = train[:, sensor_id].flatten()
        val_sensor = val[:, sensor_id].flatten()
        test_sensor = test[:, sensor_id].flatten()
        
        scaler = MinMaxScaler()
        train_norm = scaler.fit_transform(train_sensor.reshape(-1, 1)).flatten()
        val_norm = scaler.transform(val_sensor.reshape(-1, 1)).flatten()
        test_norm = scaler.transform(test_sensor.reshape(-1, 1)).flatten()
        
        train_tensor = torch.tensor(train_norm, dtype=torch.float32).unsqueeze(1)
        val_tensor = torch.tensor(val_norm, dtype=torch.float32).unsqueeze(1)
        test_tensor = torch.tensor(test_norm, dtype=torch.float32).unsqueeze(1)

        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            output = model(train_tensor.unsqueeze(0))
            loss = loss_fn(output.squeeze(), train_tensor.squeeze())
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            predictions = model(val_tensor.unsqueeze(0)).squeeze().numpy()
            predictions_desnorm = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        val_mse, val_rmse, val_mae = evaluate_model(val_sensor, predictions_desnorm[:len(val_sensor)])
        test_mse, test_rmse, test_mae = evaluate_model(test_sensor, predictions_desnorm[:len(test_sensor)])
        val_mape = mape(val_sensor, predictions_desnorm[:len(val_sensor)])
        test_mape = mape(test_sensor, predictions_desnorm[:len(test_sensor)])
        
        results.append(pd.DataFrame([{
            'sensor_id': sensor_id,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'val_rmse': val_rmse,
            'test_rmse': test_rmse,
            'val_mape': val_mape,
            'test_mape': test_mape,
            'val_mae': val_mae,
            'test_mae': test_mae
        }]))
    
    final_results = pd.concat(results, ignore_index=True)
    final_results.to_csv(results_file, index=False)
    print(f"Resultados salvos em {results_file}.")

# Executar o pipeline
file_path = 'PEMS04.npz'
results_file = 'sensor_results_transformer.csv'

pipeline_all_sensors(file_path, results_file)
