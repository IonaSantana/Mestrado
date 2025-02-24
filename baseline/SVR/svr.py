from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import os

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
        X.append(data[i - lags:i, :].flatten())  # Achatar as sequências em vetores
        y.append(data[i, :])  # Mantém todas as saídas
    return np.array(X), np.array(y)

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mse, mae, rmse

def mape(y_true, y_pred):
    epsilon = 1e-10
    percentage_error = np.abs((y_true - y_pred) / (y_true + epsilon))
    return 100 * np.mean(percentage_error)

def pipeline_all_sensors(file_path, lags_list, results_file, train_ratio=0.6, val_ratio=0.2):
    data = load_npz_data(file_path)
    train, val, test = split_data(data, train_ratio, val_ratio)

    results = []
    scaler_X, scaler_y = StandardScaler(), StandardScaler()

    for lags in lags_list:
        print(f"\nTestando SVR com {lags} lags...")
        try:
            X_train, y_train = create_sequences(train, lags)
            X_val, y_val = create_sequences(val, lags)
            X_test, y_test = create_sequences(test, lags)

            # Normalizar os dados
            X_train = scaler_X.fit_transform(X_train)
            y_train = scaler_y.fit_transform(y_train)
            X_val = scaler_X.transform(X_val)
            y_val = scaler_y.transform(y_val)
            X_test = scaler_X.transform(X_test)
            y_test = scaler_y.transform(y_test)

            # Criar e treinar o modelo MultiOutputRegressor
            base_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            model = MultiOutputRegressor(base_model)
            model.fit(X_train, y_train)

            # Fazer previsões
            predictions_val = scaler_y.inverse_transform(model.predict(X_val))
            predictions_test = scaler_y.inverse_transform(model.predict(X_test))
            y_val = scaler_y.inverse_transform(y_val)
            y_test = scaler_y.inverse_transform(y_test)

            # Avaliar o modelo
            val_mse, val_mae, val_rmse = evaluate_model(y_val, predictions_val)
            test_mse, test_mae, test_rmse = evaluate_model(y_test, predictions_test)
            val_mape = mape(y_val, predictions_val)
            test_mape = mape(y_test, predictions_test)

            result = pd.DataFrame([{
                'lags': lags,
                'val_mse': val_mse,
                'test_mse': test_mse,
                'val_mae': val_mae,
                'test_mae': test_mae,
                'val_rmse': val_rmse,
                'test_rmse': test_rmse,
                'val_mape': val_mape,
                'test_mape': test_mape,
            }])
            results.append(result)

        except Exception as e:
            print(f"Erro ao processar SVR com {lags} lags: {e}")

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
    results_file='sensor_results_SVR.csv'
)
