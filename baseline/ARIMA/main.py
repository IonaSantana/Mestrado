import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import os

# 1. Carregar os dados do arquivo .npz
def load_npz_data(file_path):
    data = np.load(file_path)
    return data['data'][:, :, 0]  # Selecionar apenas o atributo de fluxo

# 2. Dividir os dados em treino, validação e teste
def split_data(data, train_ratio=0.6, val_ratio=0.2):
    n = data.shape[0]  # Número de timestamps
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train, val, test = data[:train_end, :], data[train_end:val_end, :], data[val_end:, :]
    return train, val, test

# 3. Treinar o modelo ARIMA
def train_arima(train, order):
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    return model_fit

# 4. Avaliar o modelo
def evaluate_model(model, data):
    predictions = model.predict(start=0, end=len(data) - 1)
    mse = mean_squared_error(data, predictions)
    mae = mean_absolute_error(data, predictions)
    return mse, mae, predictions

def mape(y_true, y_pred):
    epsilon = 1e-10
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100 * np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), epsilon, None)))

# 5. Pipeline para todos os sensores e várias ordens
def pipeline_all_sensors(file_path, orders, results_file, train_ratio=0.6, val_ratio=0.2):
    # Carregar os dados
    data = load_npz_data(file_path)
    num_sensors = data.shape[1]  # Número de sensores
    
    # Dividir os dados
    train, val, test = split_data(data, train_ratio, val_ratio)
    
    # Inicializar o scaler
    scaler = MinMaxScaler()
    
    results = []
    
    # Inicializar as listas para armazenar os erros médios
    all_val_mse = []
    all_test_mse = []
    all_val_mae = []
    all_test_mae = []
    all_val_mape = []
    all_test_mape = []
    
    # Loop sobre todos os sensores
    for sensor_id in range(num_sensors):
        print(f"Treinando para o Sensor {sensor_id+1}/{num_sensors}...")

        # Dados do sensor
        train_sensor = train[:, sensor_id].flatten()
        val_sensor = val[:, sensor_id].flatten()
        test_sensor = test[:, sensor_id].flatten()

        # Normalizar os dados de treino, validação e teste
        train_sensor_norm = scaler.fit_transform(train_sensor.reshape(-1, 1)).flatten()
        val_sensor_norm = scaler.transform(val_sensor.reshape(-1, 1)).flatten()
        test_sensor_norm = scaler.transform(test_sensor.reshape(-1, 1)).flatten()

        # Testar para várias ordens de ARIMA
        for order in orders:
            print(f"  Testando ordem {order} para o Sensor {sensor_id}...")
            
            # Treinar o modelo com a ordem fixa
            model = train_arima(train_sensor_norm, order)
            
            # Calcular MSE no treinamento usando os valores ajustados
            train_mse = mean_squared_error(train_sensor_norm, model.fittedvalues)
            
            # Avaliar no conjunto de validação
            val_mse, val_mae, val_predictions = evaluate_model(model, val_sensor_norm)
            val_mape = mape(val_sensor_norm, val_predictions)
            
            # Avaliar no conjunto de teste
            test_mse, test_mae, test_predictions = evaluate_model(model, test_sensor_norm)
            test_mape = mape(test_sensor_norm, test_predictions)
            
            # Desnormalizar as previsões
            val_predictions_desnorm = scaler.inverse_transform(val_predictions.reshape(-1, 1)).flatten()
            test_predictions_desnorm = scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()

            # Adicionar os erros dos sensores às listas para a média final
            all_val_mse.append(mean_squared_error(val_sensor, val_predictions_desnorm))
            all_test_mse.append(mean_squared_error(test_sensor, test_predictions_desnorm))
            all_val_mae.append(mean_absolute_error(val_sensor, val_predictions_desnorm))
            all_test_mae.append(mean_absolute_error(test_sensor, test_predictions_desnorm))
            all_val_mape.append(mape(val_sensor, val_predictions_desnorm))
            all_test_mape.append(mape(test_sensor, test_predictions_desnorm))
            
            # Criar o DataFrame com os resultados do sensor e da ordem
            result = pd.DataFrame([{
                'sensor_id': sensor_id,
                'order': order,
                'train_mse': train_mse,
                'val_mse': all_val_mse[-1],
                'test_mse': all_test_mse[-1],
                'val_mape': val_mape,
                'test_mape': test_mape,
                'val_mae': all_val_mae[-1],
                'test_mae': all_test_mae[-1]
            }])
            
            # Adicionar aos resultados totais
            results.append(result)
    
    # Calcular a média dos erros de validação e teste por todos os sensores
    avg_val_mse = np.mean(all_val_mse)
    avg_test_mse = np.mean(all_test_mse)
    avg_val_mae = np.mean(all_val_mae)
    avg_test_mae = np.mean(all_test_mae)
    avg_val_mape = np.mean(all_val_mape)
    avg_test_mape = np.mean(all_test_mape)

    # Adicionar essas médias aos resultados finais
    final_results = pd.concat(results, ignore_index=True)
    avg_result = pd.DataFrame([{
        'sensor_id': 'All Sensors',
        'order': 'All Orders',
        'train_mse': np.mean([result['train_mse'].iloc[0] for result in results]),
        'val_mse': avg_val_mse,
        'test_mse': avg_test_mse,
        'val_mape': avg_val_mape,
        'test_mape': avg_test_mape,
        'val_mae': avg_val_mae,
        'test_mae': avg_test_mae
    }])

    final_results = pd.concat([final_results, avg_result], ignore_index=True)

    # Imprimir a média dos erros de teste para cada métrica
    print(f"\nMédia das métricas de teste para cada configuração:")
    print(f"  - Validação MAE: {avg_val_mae:.4f}")
    print(f"  - Teste MAE: {avg_test_mae:.4f}")
    print(f"  - Validação MSE: {avg_val_mse:.4f}")
    print(f"  - Teste MSE: {avg_test_mse:.4f}")
    print(f"  - Validação RMSE: {np.sqrt(avg_val_mse):.4f}")
    print(f"  - Teste RMSE: {np.sqrt(avg_test_mse):.4f}")
    print(f"  - Validação MAPE: {avg_val_mape:.4f}")
    print(f"  - Teste MAPE: {avg_test_mape:.4f}")
    
    # Salvar os resultados em um arquivo CSV
    if os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        final_results = pd.concat([existing_results, final_results], ignore_index=True)
    
    final_results.to_csv(results_file, index=False)
    print(f"Resultados salvos em {results_file}.")


# Executar o pipeline
file_path = 'PEMS04.npz'  # Substitua pelo caminho do arquivo
results_file = 'sensor_results_arima.csv'  # Arquivo para salvar os resultados

# Definir uma lista de ordens para ARIMA
orders = [
    (2, 1, 2),  # Ordem 4
]  # Você pode adicionar ou remover ordens conforme necessário

# Executar para todos os sensores com várias ordens
pipeline_all_sensors(file_path, orders, results_file)
