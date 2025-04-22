import torch
import torch.nn as nn

class LSTMCSI(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim, horizon):
        super(LSTMCSI, self).__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * output_dim)
        )
        self.output_dim = output_dim

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim]
        out = self.decoder(lstm_out[:,-1,:])  # [batch_size, horizon * output_dim]
        out = out.view(-1, self.horizon, self.output_dim)  # [batch_size, horizon, output_dim]
        return out


# Example instantiation
# if __name__ == '__main__':
#     model = LSTMCSI(
#         input_dim=128,
#         hidden_dim=256,
#         num_layers=2,
#         dropout=0.2,
#         output_dim=128,
#         horizon=3
#     )

#     dummy_input = torch.rand(32, 50, 128)  # batch_size=32, seq_len=50, feature_dim=128
#     output = model(dummy_input)
#     print(output.shape)  # Expected shape: [32, 3, 128]