import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import VAR
import os
import traceback
from sklearn.exceptions import NotFittedError

# Função para carregar dados
def load_npz_data(file_path):
    try:
        data = np.load(file_path)
        print(f"Chaves no arquivo .npz: {data.files}")
        return data['data'][:, :, 0]  # Selecionar apenas o atributo de fluxo
    except KeyError as e:
        raise ValueError(f"Chave 'data' não encontrada no arquivo {file_path}") from e
    except Exception as e:
        raise ValueError(f"Erro ao carregar o arquivo {file_path}: {e}") from e

# Função para dividir os dados
def split_data(data, train_ratio=0.6, val_ratio=0.2):
    n = data.shape[0]
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train, val, test = data[:train_end, :], data[train_end:val_end, :], data[val_end:, :]
    return train, val, test

# Função para limpar e escalar os dados
def clean_data(data, scaler=None):
    # Substituir NaNs por interpolação linear
    df = pd.DataFrame(data)
    df = df.interpolate(method='linear', axis=0)
    df = df.fillna(method='bfill').fillna(method='ffill')  # Preenchimento para trás e para frente
    data = df.to_numpy()

    # Escalar os dados
    if scaler:
        scaler.fit(data)  # Ajustar o escalador com os dados
        data = scaler.transform(data)

    return data

# Função para avaliar o modelo
def evaluate_model(y_true, y_pred, scaler=None):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    if scaler:
        try:
            y_true = scaler.inverse_transform(y_true)
            y_pred = scaler.inverse_transform(y_pred)
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
        except NotFittedError as e:
            print(f"Erro: {e}. Certifique-se de ajustar o escalador antes de usar.")
    return mse, mae, rmse

# Função para calcular MAPE
def mape(y_true, y_pred):
    epsilon = 1e-10
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100 * np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), epsilon, None)))

# Função para validar dados
def validate_data(data, label):
    if np.isnan(data).any():
        print(f"[Aviso] {label} contém valores NaN.")
    if np.isinf(data).any():
        print(f"[Aviso] {label} contém valores infinitos.")
    if np.max(data) > 1e6 or np.min(data) < -1e6:
        print(f"[Aviso] {label} contém valores extremos (fora do intervalo [-1e6, 1e6]).")

# Atualização no pipeline principal
def pipeline_all_sensors(file_path, lags_list, results_file, train_ratio=0.6, val_ratio=0.2):
    try:
        data = load_npz_data(file_path)
        print(f"Forma inicial dos dados: {data.shape}")

        scaler = MinMaxScaler()
        data = clean_data(data, scaler)

        # Validação inicial dos dados
        validate_data(data, "Dados limpos")

        train, val, test = split_data(data, train_ratio, val_ratio)
        print(f"Formas dos conjuntos: train={train.shape}, val={val.shape}, test={test.shape}")

        results = []

        for lags in lags_list:
            print(f"\nTestando VAR com {lags} lags...")
            try:
                if train.shape[0] <= lags:
                    raise ValueError(f"Número de lags ({lags}) maior ou igual ao número de amostras no conjunto de treino ({train.shape[0]}).")

                # Validar dados de treino antes do ajuste
                validate_data(train, "Conjunto de treino")

                model = VAR(train)
                model_fitted = model.fit(lags)

                predictions_val = model_fitted.forecast(y=train[-lags:], steps=val.shape[0])
                predictions_test = model_fitted.forecast(y=np.vstack((train, val))[-lags:], steps=test.shape[0])

                # Validar previsões
                predictions_val = np.nan_to_num(predictions_val, nan=0.0, posinf=1e6, neginf=-1e6)
                predictions_test = np.nan_to_num(predictions_test, nan=0.0, posinf=1e6, neginf=-1e6)
                validate_data(predictions_val, "Previsões de validação")
                validate_data(predictions_test, "Previsões de teste")

                val_mse, val_mae, val_rmse = evaluate_model(val, predictions_val, scaler)
                test_mse, test_mae, test_rmse = evaluate_model(test, predictions_test, scaler)
                val_mape = mape(val, predictions_val)
                test_mape = mape(test, predictions_test)

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
                print(f"Erro ao processar VAR com {lags} lags: {e}")
                traceback.print_exc()

        if results:
            final_results = pd.concat(results, ignore_index=True)
            if os.path.exists(results_file):
                try:
                    existing_results = pd.read_csv(results_file)
                    final_results = pd.concat([existing_results, final_results], ignore_index=True)
                except Exception as e:
                    print(f"Erro ao carregar resultados existentes: {e}")
                    traceback.print_exc()

            final_results.to_csv(results_file, index=False)
            print(final_results)
            print(f"Resultados salvos em {results_file}.")
        else:
            print("Nenhum resultado para salvar.")

    except Exception as e:
        print(f"Erro na execução do pipeline: {e}")
        traceback.print_exc()

# Funções auxiliares como load_npz_data, clean_data, evaluate_model e mape permanecem as mesmas.


# Executar o pipeline
pipeline_all_sensors(
    file_path='PEMS04.npz', 
    lags_list=[3, 5, 7], 
    results_file='sensor_results_VAR.csv'
)
