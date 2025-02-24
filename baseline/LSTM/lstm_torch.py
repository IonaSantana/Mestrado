import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Carregar dados do arquivo .npz
def load_npz_data(file_path):
    data = np.load(file_path)
    return data['data'][:, :, 0]  # Selecionar apenas o atributo de fluxo

# Dividir os dados em conjuntos de treino, validação e teste
def split_data(data, train_ratio=0.6, val_ratio=0.2):
    n = data.shape[0]
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train, val, test = data[:train_end, :], data[train_end:val_end, :], data[val_end:, :]
    return train, val, test

# Criar sequências para entrada na LSTM
def create_sequences(data, lags):
    X, y = [], []
    for i in range(lags, data.shape[0]):
        X.append(data[i - lags:i, :])
        y.append(data[i, :])
    return np.array(X), np.array(y)

# Dataset personalizado para PyTorch
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# Funções de avaliação
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mse, mae, rmse

# Função MAPE (Mean Absolute Percentage Error)
def mape(y_true, y_pred):
    epsilon = 1e-10  # Para evitar divisão por zero
    percentage_error = np.abs((y_true - y_pred) / (y_true + epsilon))  # Erro percentual
    return 100 * np.mean(percentage_error)  # Retorna o MAPE como porcentagem

# Pipeline principal
def pipeline_all_sensors(file_path, lags_list, results_file, train_ratio=0.6, val_ratio=0.2, epochs=10, batch_size=32, hidden_dim=64):
    data = load_npz_data(file_path)

    train, val, test = split_data(data, train_ratio, val_ratio)

    results = []

    for lags in lags_list:
        print(f"\nTestando LSTM com {lags} lags...")
        try:
            # Criar sequências
            X_train, y_train = create_sequences(train, lags)
            X_val, y_val = create_sequences(val, lags)
            X_test, y_test = create_sequences(test, lags)

            # Criar datasets e dataloaders
            train_dataset = TimeSeriesDataset(X_train, y_train)
            val_dataset = TimeSeriesDataset(X_val, y_val)
            test_dataset = TimeSeriesDataset(X_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Definir modelo, loss e otimizador
            input_dim = X_train.shape[2]
            output_dim = y_train.shape[1]
            model = LSTMModel(input_dim, hidden_dim, output_dim)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Treinar o modelo
            model.train()
            for epoch in range(epochs):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()

                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

            # Avaliar o modelo
            model.eval()
            with torch.no_grad():
                val_predictions = []
                for X_batch, _ in val_loader:
                    val_predictions.append(model(X_batch).numpy())
                val_predictions = np.vstack(val_predictions)

                test_predictions = []
                for X_batch, _ in test_loader:
                    test_predictions.append(model(X_batch).numpy())
                test_predictions = np.vstack(test_predictions)

            val_mse, val_mae, val_rmse = evaluate_model(y_val.flatten(), val_predictions.flatten())
            test_mse, test_mae, test_rmse = evaluate_model(y_test.flatten(), test_predictions.flatten())
            val_mape = mape(y_val.flatten(), val_predictions.flatten())  # Alterado para MAPE
            test_mape = mape(y_test.flatten(), test_predictions.flatten())  # Alterado para MAPE

            result = pd.DataFrame([{
                'lags': lags,
                'val_mse': val_mse,
                'test_mse': test_mse,
                'val_mae': val_mae,
                'test_mae': test_mae,
                'val_rmse': val_rmse,
                'test_rmse': test_rmse,
                'val_mape': val_mape,  # Alterado para MAPE
                'test_mape': test_mape,  # Alterado para MAPE
            }])
            results.append(result)

        except Exception as e:
            print(f"Erro ao processar LSTM com {lags} lags: {e}")

    # Salvar resultados
    final_results = pd.concat(results, ignore_index=True)
    if os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        final_results = pd.concat([existing_results, final_results], ignore_index=True)
    final_results.to_csv(results_file, index=False)
    print(f"Resultados salvos em {results_file}.")

# Executar o pipeline
pipeline_all_sensors(
    file_path='PEMS04.npz',
    lags_list=[ 7],
    results_file='sensor_results_LSTM_PyTorch.csv',
    epochs=200,
    batch_size=64,
    hidden_dim=64
)