import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


class Transformer1D(nn.Module):
    def __init__(self, input_dim=307, d_model=64, num_heads=8, num_layers=3):
        super(Transformer1D, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)  # Mapeia de 307 para 64
        self.positional_encoding = PositionalEncoding(d_model)  # Adiciona posição
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, input_dim)  # Volta de 64 para 307

    def forward(self, x):
        print(f"Input shape: {x.shape}")  # Para debug: (1, 16992, 307)
        
        x = self.input_proj(x)  # (1, 16992, 307) -> (1, 16992, 64)
        print(f"After projection: {x.shape}")
        
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        
        x = self.output_layer(x)  # (1, 16992, 64) -> (1, 16992, 307)
        print(f"Output shape: {x.shape}")  # Para debug

        return x
