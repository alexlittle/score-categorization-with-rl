import torch
import torch.nn as nn


class LSTM_DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(LSTM_DQN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):

        out, hidden = self.lstm(x, hidden)

        # check if there's batch size or not
        if out.dim() == 3:
            out = out[:, -1, :]
        elif out.dim() == 2:
            out = out[-1, :]

        q_values = self.fc(out)
        return q_values, hidden

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)