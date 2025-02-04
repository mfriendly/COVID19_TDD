import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.torch_dtype = torch.float64 if args.dtype == "double" else torch.float32
        self.in_channels = args.in_channels
        self.hidden_channels = args.hidden_size
        self.output_size = args.output_size
        self.device = args.device
        self.dropout = args.dropout
        self.lstm = nn.LSTM(
            in_channels=self.in_channels,
            hidden_size=self.hidden_channels,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout,
        )
        # Output projection
        self.output_layer = nn.Linear(self.hidden_channels, 1)

    def forward(self, x, target_cl=None, task_level=42, global_step=None):
        x.shape[0]
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        # Initialize output sequence
        outputs = []
        # Use last hidden state for prediction
        h_t = hidden[-1]
        # Generate multi-horizon predictions
        for _ in range(task_level):
            # Generate one-step prediction
            out = self.output_layer(h_t)
            outputs.append(out.unsqueeze(1))
            # Use prediction as next input
            _, (h_t, _) = self.lstm(out.unsqueeze(1), (hidden, cell))
            h_t = h_t[-1]
        # Combine all predictions
        outputs = torch.cat(outputs, dim=1)
        return outputs


class SimpleGRU(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.torch_dtype = torch.float64 if args.dtype == "double" else torch.float32
        self.in_channels = args.in_channels
        self.hidden_channels = args.hidden_size
        self.output_size = args.output_size
        self.device = args.device
        self.dropout = args.dropout
        self.gru = nn.GRU(
            in_channels=self.in_channels,
            hidden_size=self.hidden_channels,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout,
        )
        # Output projection
        self.output_layer = nn.Linear(self.hidden_channels, 1)

    def forward(self, x, target_cl=None, task_level=42, global_step=None):
        x.shape[0]
        # Pass through GRU
        gru_out, hidden = self.gru(x)
        # Initialize output sequence
        outputs = []
        # Use last hidden state for prediction
        h_t = hidden[-1]
        # Generate multi-horizon predictions
        for _ in range(task_level):
            # Generate one-step prediction
            out = self.output_layer(h_t)
            outputs.append(out.unsqueeze(1))
            # Use prediction as next input
            _, h_t = self.gru(out.unsqueeze(1), hidden)
            h_t = h_t[-1]
        # Combine all predictions
        outputs = torch.cat(outputs, dim=1)
        return outputs
