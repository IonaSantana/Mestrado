import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. Carregar os dados do arquivo .npz
def load_npz_data(file_path):
    data = np.load(file_path)
    return data['data'][:, :, 0]  # Selecionar apenas o atributo de fluxo

def split_and_normalize_data(data, train_ratio=0.6, val_ratio=0.2):
    n = data.shape[0]
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = data[:train_end, :]
    val_data = data[train_end:val_end, :]
    test_data = data[val_end:, :]
    
    # Aplicar log nas séries antes da normalização
    train_data = np.log1p(train_data)
    val_data = np.log1p(val_data)
    test_data = np.log1p(test_data)
    
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)
    
    return train_data, val_data, test_data, scaler

