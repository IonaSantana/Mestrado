import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import os

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

# 3. Aplicar Convolução 1D
def apply_convolution1d(train, kernel_size=3):
    """Aplica uma convolução 1D simples para suavização."""
    kernel = np.ones(kernel_size) / kernel_size
    conv_result = np.convolve(train, kernel, mode='same')
    return conv_result

# 4. Avaliar o modelo
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, rmse, mae

def mape(y_true, y_pred, epsilon=1e-5):
    mask = np.abs(y_true) > epsilon
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

# 5. Pipeline para todos os sensores usando Convolução 1D
def pipeline_all_sensors(file_path, kernel_sizes, results_file, train_ratio=0.6, val_ratio=0.2):
    data = load_npz_data(file_path)
    num_sensors = data.shape[1]
    train, val, test = split_data(data, train_ratio, val_ratio)
    results = []
    best_kernel, best_val_mae = None, float('inf')

    for kernel_size in kernel_sizes:
        print(f"\nTestando kernel_size = {kernel_size}...")
        all_val_mae = []
        all_test_mae = []

        for sensor_id in range(num_sensors):
            train_sensor = train[:, sensor_id].flatten()
            val_sensor = val[:, sensor_id].flatten()
            test_sensor = test[:, sensor_id].flatten()
            
            scaler = MinMaxScaler()
            train_norm = scaler.fit_transform(train_sensor.reshape(-1, 1)).flatten()
            val_norm = scaler.transform(val_sensor.reshape(-1, 1)).flatten()
            test_norm = scaler.transform(test_sensor.reshape(-1, 1)).flatten()
            
            predictions = apply_convolution1d(train_norm, kernel_size)
            predictions = np.pad(predictions, (len(train_sensor) - len(predictions), 0), 'edge')
            predictions = np.nan_to_num(predictions, nan=0)
            predictions_desnorm = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            
            val_mse, val_rmse, val_mae = evaluate_model(val_sensor, predictions_desnorm[:len(val_sensor)])
            test_mse, test_rmse, test_mae = evaluate_model(test_sensor, predictions_desnorm[:len(test_sensor)])
            val_mape = mape(val_sensor, predictions_desnorm[:len(val_sensor)])
            test_mape = mape(test_sensor, predictions_desnorm[:len(test_sensor)])
            
            all_val_mae.append(val_mae)
            all_test_mae.append(test_mae)
            
            results.append(pd.DataFrame([{
                'sensor_id': sensor_id,
                'kernel_size': kernel_size,
                'val_mse': val_mse,
                'test_mse': test_mse,
                'val_rmse': val_rmse,
                'test_rmse': test_rmse,
                'val_mape': val_mape,
                'test_mape': test_mape,
                'val_mae': val_mae,
                'test_mae': test_mae
            }]))
        
        avg_val_mae = np.mean(all_val_mae)
        if avg_val_mae < best_val_mae:
            best_kernel, best_val_mae = kernel_size, avg_val_mae
    
    final_results = pd.concat(results, ignore_index=True)
    print(f"\nMelhor kernel_size: {best_kernel} com Validação MAE: {best_val_mae:.4f}")
    final_results.to_csv(results_file, index=False)
    print(f"Resultados salvos em {final_results}.")

# Executar o pipeline
file_path = 'PEMS04.npz'
results_file = 'sensor_results_conv1.csv'
kernel_sizes = [3, 5, 7, 9]

pipeline_all_sensors(file_path, kernel_sizes, results_file)
