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
    n = data.shape[0]  # Número de timestamps
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train, val, test = data[:train_end, :], data[train_end:val_end, :], data[val_end:, :]
    return train, val, test

# 3. Aplicar Média Móvel Exponencial (EMA)
def exponential_moving_average(train, alpha=0.1):
    """Calcular a Média Móvel Exponencial com fator de suavização alpha."""
    ema = pd.Series(train).ewm(alpha=alpha, adjust=False).mean()
    return ema.values

# 4. Avaliar o modelo
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # Calculando RMSE
    mae = mean_absolute_error(y_true, y_pred)
    return mse, rmse, mae

def smape(y_true, y_pred):
    """Calcula o SMAPE entre os valores reais e previstos, evitando NaN devido a divisão por zero."""
    epsilon = 1e-10
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon))

def mape(y_true, y_pred):
    epsilon = 1e-5
    # Filtra os valores de y_true menores que epsilon
    mask = np.abs(y_true) > epsilon
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return 100 * np.mean(np.abs((y_true - y_pred) / y_true))

# 5. Pipeline para todos os sensores usando EMA
def pipeline_all_sensors(file_path, alpha_values, results_file, train_ratio=0.6, val_ratio=0.2):
    # Carregar os dados
    data = load_npz_data(file_path)
    num_sensors = data.shape[1]  # Número de sensores
    
    # Dividir os dados
    train, val, test = split_data(data, train_ratio, val_ratio)
    
    # Normalizar os dados (usar MinMaxScaler)
    # scaler = MinMaxScaler()
    # train_scaled = scaler.fit_transform(train)
    # val_scaled = scaler.transform(val)
    # test_scaled = scaler.transform(test)
    
    results = []
    
    # Inicializar as listas para armazenar os erros médios
    all_val_mse = []
    all_test_mse = []
    all_val_rmse = []
    all_test_rmse = []
    all_val_mae = []
    all_test_mae = []
    all_val_mape = []
    all_test_mape = []
    
    # Variáveis para armazenar a melhor configuração e resultados
    best_alpha = None
    best_val_mae = float('inf')
    best_test_mae = float('inf')
    best_val_mse = float('inf')
    best_test_mse = float('inf')
    best_val_mape = float('inf')
    best_test_mape = float('inf')
    
    # Loop para testar diferentes valores de alpha
    for alpha in alpha_values:
        print(f"\nTestando alpha = {alpha}...")

        # Loop sobre todos os sensores
        # Loop sobre todos os sensores
        for sensor_id in range(num_sensors):
            print(f"  Treinando para o Sensor {sensor_id+1}/{num_sensors}...")

            # Dados do sensor
            train_sensor = train[:, sensor_id].flatten()
            val_sensor = val[:, sensor_id].flatten()
            test_sensor = test[:, sensor_id].flatten()
            
            # Inicializar o scaler para o sensor específico
            scaler = MinMaxScaler()

            # Normalizar os dados de treino, validação e teste para o sensor específico
            train_sensor_norm = scaler.fit_transform(train_sensor.reshape(-1, 1)).flatten()
            val_sensor_norm = scaler.transform(val_sensor.reshape(-1, 1)).flatten()
            test_sensor_norm = scaler.transform(test_sensor.reshape(-1, 1)).flatten()
            
            # Aplicar Média Móvel Exponencial para previsão
            predictions = exponential_moving_average(train_sensor_norm, alpha)

            # Ajustar o tamanho das previsões para o conjunto de validação e teste
            predictions = np.pad(predictions, (len(train_sensor) - len(predictions), 0), 'edge')

            # Substituir NaN com 0 nas previsões (caso ocorram)
            predictions = np.nan_to_num(predictions, nan=0)

            predictions_reshaped = predictions.reshape(-1, 1)
            predictions_desnorm = scaler.inverse_transform(predictions_reshaped).flatten()

            # Calcular as métricas de erro na escala original
            val_mse, val_rmse, val_mae = evaluate_model(val_sensor, predictions_desnorm[:len(val_sensor)])
            val_mape = mape(val_sensor, predictions_desnorm[:len(val_sensor)])

            test_mse, test_rmse, test_mae = evaluate_model(test_sensor, predictions_desnorm[:len(test_sensor)])
            test_mape = mape(test_sensor, predictions_desnorm[:len(test_sensor)])

            # Adicionar os erros dos sensores às listas para a média final
            all_val_mse.append(val_mse)
            all_test_mse.append(test_mse)
            all_val_rmse.append(val_rmse)
            all_test_rmse.append(test_rmse)
            all_val_mae.append(val_mae)
            all_test_mae.append(test_mae)
            all_val_mape.append(val_mape)
            all_test_mape.append(test_mape)

            # Criar o DataFrame com os resultados do sensor e do alpha
            result = pd.DataFrame([{
                'sensor_id': sensor_id,
                'alpha': alpha,
                'val_mse': val_mse,
                'test_mse': test_mse,
                'val_rmse': val_rmse,
                'test_rmse': test_rmse,
                'val_mape': val_mape,
                'test_mape': test_mape,
                'val_mae': val_mae,
                'test_mae': test_mae
            }])

            # Adicionar aos resultados totais
            results.append(result)
        
        # Calcular a média dos erros de validação e teste para este alpha
        avg_val_mae = np.mean(all_val_mae[-num_sensors:])
        avg_test_mae = np.mean(all_test_mae[-num_sensors:])
        avg_val_mse = np.mean(all_val_mse[-num_sensors:])
        avg_test_mse = np.mean(all_test_mse[-num_sensors:])
        avg_val_rmse = np.mean(all_val_rmse[-num_sensors:])
        avg_test_rmse = np.mean(all_test_rmse[-num_sensors:])
        avg_val_mape = np.mean(all_val_mape[-num_sensors:])
        avg_test_mape = np.mean(all_test_mape[-num_sensors:])
        
        # Verificar se este alpha tem o melhor desempenho
        if avg_val_mse < best_val_mse:
            best_alpha = alpha
            best_val_mae = avg_val_mae
            best_test_mae = avg_test_mae
            best_val_mse = avg_val_mse
            best_test_mse = avg_test_mse
            best_val_rmse = avg_val_rmse
            best_test_rmse = avg_test_rmse
            best_val_mape = avg_val_mape
            best_test_mape = avg_test_mape

    # Adicionar os resultados finais para o melhor alpha
    final_results = pd.concat(results, ignore_index=True)

    # Exibir os resultados do melhor alpha
    print(f"\nMelhor alpha: {best_alpha}")
    print(f"  - Validação MAE: {best_val_mae:.4f}")
    print(f"  - Teste MAE: {best_test_mae:.4f}")
    print(f"  - Validação MSE: {best_val_mse:.4f}")
    print(f"  - Teste MSE: {best_test_mse:.4f}")
    print(f"  - Validação RMSE: {best_val_rmse:.4f}")
    print(f"  - Teste RMSE: {best_test_rmse:.4f}")
    print(f"  - Validação MAPE: {best_val_mape:.4f}")
    print(f"  - Teste MAPE: {best_test_mape:.4f}")
    
    # Salvar os resultados em um arquivo CSV
    if os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        final_results = pd.concat([existing_results, final_results], ignore_index=True)
    
    final_results.to_csv(results_file, index=False)
    print(f"Resultados salvos em {results_file}.")


#
# Executar o pipeline
file_path = 'PEMS04.npz'  # Substitua pelo caminho do arquivo
results_file = 'sensor_results_ema.csv'  # Arquivo para salvar os resultados

# Definir uma lista de valores de alpha para a Média Móvel Exponencial
alpha_values = [0.05, 0.1, 0.2, 0.01, 0.001]  # Diferentes valores de alpha

# Executar para todos os sensores com diferentes valores de alpha
pipeline_all_sensors(file_path, alpha_values, results_file)