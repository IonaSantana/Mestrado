import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def load_npz_data(file_path):
    data = np.load(file_path)
    return data['data'][:, :, 0]  # Selecionar apenas o atributo de fluxo

def split_data(data, train_ratio=0.6, val_ratio=0.2):
    n = data.shape[0]
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train, val, test = data[:train_end, :], data[train_end:val_end, :], data[val_end:, :]
    return train, val, test

def create_sequences(data, lags):
    X, y = [], []
    for i in range(lags, data.shape[0]):
        X.append(data[i - lags:i, :])
        y.append(data[i, :])
    return np.array(X), np.array(y)

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mse, mae, rmse

def smape(y_true, y_pred):
    epsilon = 1e-10
    numerator = 2 * np.abs(y_pred - y_true)
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    smape_values = numerator / denominator
    smape_values = np.nan_to_num(smape_values, nan=0.0, posinf=0.0, neginf=0.0)
    return 100 * np.mean(smape_values)

def mape(y_true, y_pred):
    epsilon = 1e-10  # Para evitar divisão por zero
    percentage_error = np.abs((y_true - y_pred) / (y_true + epsilon))  # Erro percentual
    return 100 * np.mean(percentage_error)  # Retorna o MAPE como porcentagem

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape, return_sequences=False),
        Dense(input_shape[-1])  # Número de saídas igual ao número de atributos
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def pipeline_all_sensors(file_path, lags_list, results_file, train_ratio=0.6, val_ratio=0.2, epochs=10, batch_size=32):
    data = load_npz_data(file_path)

    train, val, test = split_data(data, train_ratio, val_ratio)

    results = []

    for lags in lags_list:
        print(f"\nTestando LSTM com {lags} lags...")
        try:
            X_train, y_train = create_sequences(train, lags)
            X_val, y_val = create_sequences(val, lags)
            X_test, y_test = create_sequences(test, lags)

            model = build_lstm_model(X_train.shape[1:])
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)

            predictions_val = model.predict(X_val)
            predictions_test = model.predict(X_test)
            print(predictions_val)
            val_mse, val_mae, val_rmse = evaluate_model(y_val.flatten(), predictions_val.flatten())
            test_mse, test_mae, test_rmse = evaluate_model(y_test.flatten(), predictions_test.flatten())
            val_smape = mape(y_val.flatten(), predictions_val.flatten())
            test_smape = mape(y_test.flatten(), predictions_test.flatten())

            result = pd.DataFrame([{
                'lags': lags,
                'val_mse': val_mse,
                'test_mse': test_mse,
                'val_mae': val_mae,
                'test_mae': test_mae,
                'val_rmse': val_rmse,
                'test_rmse': test_rmse,
                'val_mape': val_smape,
                'test_mape': test_smape,
            }])
            results.append(result)

        except Exception as e:
            print(f"Erro ao processar LSTM com {lags} lags: {e}")

    final_results = pd.concat(results, ignore_index=True)
    if os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        final_results = pd.concat([existing_results, final_results], ignore_index=True)
    final_results.to_csv(results_file, index=False)
    print(f"Resultados salvos em {results_file}.")

# Executar o pipeline
pipeline_all_sensors(
    file_path='PEMS04.npz',
    lags_list=[3, 5, 7],
    results_file='sensor_results_LSTM.csv',
    epochs=50,
    batch_size=64
)
