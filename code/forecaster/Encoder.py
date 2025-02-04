import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .CustomLSTMCell import CustomLSTMCell


class Encoder(nn.Module):

    def __init__(
        self,
        args,
    ):
        super(Encoder, self).__init__()
        self.torch_dtype = torch.float64 if args.dtype == "double" else torch.float32
        if True:
            args.in_channels = args.hidden_size
        self.in_channels = args.in_channels
        self.time_size = args.past_steps
        self.hidden_channels = args.hidden_size
        self.output_size = args.output_size
        self.in_channels = self.in_channels
        self.time_channels = self.time_size
        self.hidden_channels = self.hidden_channels
        self.output_channels = self.output_size
        self.num_for_predict = args.num_for_predict
        self.dropout = args.dropout
        self.dropout_type = args.dropout_type
        self.fusion_mode = args.fusion_mode
        self.device = args.device
        self.dropout = nn.Dropout(p=self.dropout)
        self.seq_length = self.num_for_predict
        self.RNN_layer = 1
        self.RNNCell = nn.ModuleList([CustomLSTMCell(args, self.in_channels)])

    def forward(self, x, seq_length):
        """
        :param x: [batch_size, num_pred*3, x_size]
        :param seq_length: num_pred
        :return: [batch_size, num_pred, hidden_size]
        """
        batch_size, time_len, dim = x.shape
        Hidden_State = [
            self.initHidden(batch_size, self.hidden_channels)
            for _ in range(self.RNN_layer)
        ]
        Cell_State = [
            self.initHidden(batch_size, self.hidden_channels)
            for _ in range(self.RNN_layer)
        ]
        outputs = []
        hiddens = []
        for i in range(seq_length):
            input_cur = x[:, i * 1:i * 1 + 1, :]
            for j, rnn_cell in enumerate(self.RNNCell):
                cur_h = Hidden_State[j]
                cur_c = Cell_State[j]
                cur_out, (cur_h, cur_c) = rnn_cell(input_cur, (cur_h, cur_c))
                Hidden_State[j] = cur_h
                Cell_State[j] = cur_c
                input_cur = torch.tanh(cur_out)
            outputs.append(cur_out.unsqueeze(dim=1))
            hidden = torch.stack(Hidden_State, dim=1).unsqueeze(dim=2)
            hiddens.append(hidden)
        outputs = torch.cat(outputs, dim=1)
        hiddens = torch.cat(hiddens, dim=2)
        return outputs, hiddens

    def initHidden(self, batch_size, hidden_size):
        """
        Initialize the hidden state using He initialization
        :param batch_size:
        :param hidden_size:
        :return:
        """
        std = np.sqrt(2.0 / hidden_size)
        return Variable(
            torch.randn((batch_size, hidden_size), dtype=self.torch_dtype).to(
                self.device) * std)
