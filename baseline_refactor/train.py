
import pandas as pd
import torch
import torch.nn as nn
import time
from pre_processing import load_npz_data, split_and_normalize_data
from tecnicas.transformer1d import Transformer1D 
from evaluate import evaluate_model
from tqdm import tqdm  # Importando a biblioteca tqdm

def pipeline_all_sensors(file_path, model, results_file, results_file_mean, train_ratio=0.6, val_ratio=0.2, epochs=100, lr=0.001):
    # Carregar dados e pré-processamento
    data = load_npz_data(file_path)
    num_sensors = data.shape[1]
    print(f"Número de sensores: {num_sensors}")
    print(f"Forma dos dados: {data.shape}")
    
    # Divisão dos dados em treino, validação e teste
    train, val, test, scaler = split_and_normalize_data(data, train_ratio, val_ratio)

    # Inicializar o modelo, otimizador e função de perda
    model = Transformer1D(input_dim=num_sensors, d_model=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Organizar os dados em tensores para o modelo
    def prepare_tensor(data):
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    # Preparando os dados para o modelo (forma: batch_size, seq_len, input_dim)
    train_tensor = prepare_tensor(train)
    val_tensor = prepare_tensor(val)
    test_sensor = prepare_tensor(test)

    # Variáveis para monitorar o progresso do treinamento
    best_val_loss = float('inf')
    patience = 15  # Número de épocas sem melhoria antes de parar
    epochs_without_improvement = 0

    # Treinamento com tqdm
    for epoch in tqdm(range(epochs), desc="Treinamento", unit="época"):
        start_time = time.time()  # Marcar o tempo de início da época
        
        model.train()
        optimizer.zero_grad()
        output = model(train_tensor)
        loss = loss_fn(output.squeeze(), train_tensor.squeeze())
        loss.backward()
        optimizer.step()

        # Monitorando a perda de validação a cada época
        model.eval()
        with torch.no_grad():
            val_output = model(val_tensor)
            val_predictions = val_output.squeeze().numpy()

            # Ajuste no tamanho das previsões de validação
            if val_predictions.shape[0] != val.flatten().shape[0]:
                val_predictions = val_predictions[:val.flatten().shape[0]]  # Cortando para coincidir

            # print(f"val shape after flatten: {val.flatten().shape}")
            # print(f"val shape after flatten: {val.shape}")
            # print(f"val_predictions shape: {val_predictions.shape}")
            val_metrics = evaluate_model(val, val_predictions)

        # Tempo da época
        epoch_time = time.time() - start_time
        print('Val metrics _ ',val_metrics)
        print(f"Epoch {epoch+1}/{epochs}, Time: {epoch_time:.2f}s, Val Loss: {loss.item()}, Val MSE: {val_metrics[0]}, Val RMSE: {val_metrics[1]}, Val MAE: {val_metrics[2]}, Val MAPE: {val_metrics[3]}")

        # Verificando melhoria na perda de validação
        if loss.item() < best_val_loss:
            best_val_loss = loss.item()
            epochs_without_improvement = 0
            # Salvar o melhor modelo
            torch.save(model.state_dict(), 'best_model.pth')
            # Avaliação no conjunto de teste
            with torch.no_grad():
                test_output = model(test_sensor)
                test_predictions = test_output.squeeze().numpy()

                # Ajuste no tamanho das previsões de teste
                if test_predictions.shape[0] != test.flatten().shape[0]:
                    test_predictions = test_predictions[:test.flatten().shape[0]]  # Cortando para coincidir

                test_metrics = evaluate_model(test, test_predictions)
            print(f"Test MSE: {test_metrics[0]}, Test RMSE: {test_metrics[1]}, Test MAE: {test_metrics[2]}, Test MAPE: {test_metrics[3]}")
        else:
            epochs_without_improvement += 1
        
        # Parar o treinamento se não houver melhoria por 'patience' épocas
        if epochs_without_improvement >= patience:
            print(f"Parando o treinamento após {patience} épocas sem melhoria.")
            break

        # Feedback contínuo e melhoria
        print(f"Feedback da época {epoch+1}:")
        print(f"Perda de Treinamento: {loss.item()}")
        print(f"Perda de Validação: {val_metrics[0]}")
        print(f"Tempo da Época: {epoch_time:.2f}s")

    # Avaliação final no conjunto de validação
    model.eval()
    with torch.no_grad():
        val_output = model(val_tensor)
        val_predictions = val_output.squeeze().numpy()
        

        # Ajuste no tamanho das previsões de validação
        if val_predictions.shape[0] != val.flatten().shape[0]:
            val_predictions = val_predictions[:val.flatten().shape[0]]  # Cortando para coincidir

        # print(f"val shape after flatten: {val.flatten().shape}")
        # print(f"val shape after flatten: {val.shape}")
        # print(f"val_predictions shape: {val_predictions.shape}")
        val_metrics = evaluate_model(val, val_predictions)
        print(f"val shape after flatten: {val.flatten().shape}")
        print(f"val_predictions shape: {val_predictions.shape}")

    # Avaliação no conjunto de teste
    with torch.no_grad():
        test_output = model(test_sensor)
        test_predictions = test_output.squeeze().numpy()

        # Ajuste no tamanho das previsões de teste
        if test_predictions.shape[0] != test.flatten().shape[0]:
            test_predictions = test_predictions[:test.flatten().shape[0]]  # Cortando para coincidir

        test_metrics = evaluate_model(test, test_predictions)

    # Armazenar os resultados
    results = pd.DataFrame([{
        'val_mse': val_metrics[0],
        'test_mse': test_metrics[0],
        'val_rmse': val_metrics[1],
        'test_rmse': test_metrics[1],
        'val_mape': val_metrics[3],
        'test_mape': test_metrics[3],
        'val_mae': val_metrics[2],
        'test_mae': test_metrics[2]
    }])

    # Salvar os resultados
    results.to_csv(results_file, index=False)
    results_mean = results.mean()
    results_mean.to_csv(results_file_mean, index=False)

    print(f"Erro por sensor salvo em {results_file}.")
    print(f"Erro médio salvo em {results_file_mean}.")


