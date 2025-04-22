import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerCSI(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout, max_len, horizon):
        super(TransformerCSI, self).__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.time_step = max_len
        self.horizon = horizon
        self.pos_encoder = PositionalEncoding(model_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder1 = nn.Linear(model_dim, input_dim)
        self.decoder2 = nn.Sequential(
            nn.Linear(int(self.time_step), int(self.time_step)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step), self.horizon),
        )

    def forward(self, src):
        # src shape: [batch_size, sequence_length, input_dim]
        src = self.input_projection(src)  # project to model_dim
        src = self.pos_encoder(src)
        transformer_output = self.transformer_encoder(src)
        d_output = self.decoder1(transformer_output)
        d_output = d_output.permute(0,2,1)
        output = self.decoder2(d_output)  # take the last time step
        return output.permute(0,2,1).contiguous()


# if __name__ == '__main__':
#     model = TransformerCSI(
#         input_dim=128,     # Feature dimension of each timestep
#         model_dim=256,     # Internal model dimension
#         num_heads=4,       # Multi-head attention heads
#         num_layers=4,      # Number of encoder layers
#         dropout=0.1,       # Dropout rate
#         output_dim=128,    # Output dimension (e.g., same as input_dim for reconstruction)
#         max_len=50,         # Max input sequence length
#         horizon=3
#     )

#     dummy_input = torch.rand(32, 50, 128)  # batch_size=32, seq_len=50, feature_dim=128
#     output = model(dummy_input)
#     print(output.shape)  # Expected shape: [32, 3, 128]