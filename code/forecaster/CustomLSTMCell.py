import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLSTMCell(nn.Module):

    def __init__(self, args, in_channels):
        super().__init__()
        self.torch_dtype = torch.float64 if args.dtype == "double" else torch.float32
        self.in_channels = in_channels
        self.time_size = args.past_steps
        self.hidden_channels = args.hidden_size
        self.output_size = args.output_size
        self.time_channels = self.time_size
        self.hidden_channels = self.hidden_channels
        self.output_channels = self.output_size
        self.dropout_type = args.dropout_type
        self.dropout = args.dropout
        self.activation_name1 = args.activations[0]
        self.dropout_layer = nn.Dropout(self.dropout)
        self.add_skips = getattr(args, "add_skips", False)
        self.input_block = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_channels),
            nn.LayerNorm(self.hidden_channels),
            self.dropout_layer,
        )
        if self.add_skips:
            self.input_skip = nn.Linear(self.in_channels, self.hidden_channels)
            self.hidden_skip = nn.Linear(self.hidden_channels,
                                         self.hidden_channels)
            self.gate_skip = nn.Linear(self.hidden_channels * 2,
                                       self.hidden_channels)
        self.forget_gate = nn.Sequential(
            nn.Linear(self.hidden_channels * 2, self.hidden_channels),
            nn.LayerNorm(self.hidden_channels),
            self.dropout_layer,
        )
        self.input_gate = nn.Sequential(
            nn.Linear(self.hidden_channels * 2, self.hidden_channels),
            nn.LayerNorm(self.hidden_channels),
            self.dropout_layer,
        )
        self.cell_gate = nn.Sequential(
            nn.Linear(self.hidden_channels * 2, self.hidden_channels),
            nn.LayerNorm(self.hidden_channels),
            self.dropout_layer,
        )
        self.output_gate = nn.Sequential(
            nn.Linear(self.hidden_channels * 2, self.hidden_channels),
            nn.LayerNorm(self.hidden_channels),
            self.dropout_layer,
        )
        self.layerNorm = nn.LayerNorm(self.hidden_channels * 2)
        self.cell_norm = nn.LayerNorm(self.hidden_channels)
        self.hidden_norm = nn.LayerNorm(self.hidden_channels)
        self.identity = nn.Identity()
        self.nlayer_geatt3a = args.nlayer_geatt3a
        self.nlayer_geatt3b = args.nlayer_geatt3b
        from .GEAtt import GEAttention
        if self.nlayer_geatt3a > 0:
            self.GEAttRecurrent_block1 = GEAttention(args,
                                                     self.hidden_channels * 2,
                                                     8, 0.6)
        if self.nlayer_geatt3b > 0:
            self.GEAttRecurrent_block2 = GEAttention(args,
                                                     self.hidden_channels, 8,
                                                     0.6)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_in",
                                        nonlinearity="selu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.input_block.apply(init_weights)
        if self.add_skips:
            self.input_skip.apply(init_weights)
            self.hidden_skip.apply(init_weights)
            self.gate_skip.apply(init_weights)
        self.forget_gate.apply(init_weights)
        self.input_gate.apply(init_weights)
        self.cell_gate.apply(init_weights)
        self.output_gate.apply(init_weights)
        self.input_proj = nn.Linear(self.in_channels, self.hidden_channels)

    def forward(self, input, states, encoder_hidden=None):
        activation_fn = self._get_activation(self.activation_name1)
        hidden_state, cell_state = states
        x = input.squeeze(1)
        batch_size, in_channels = x.shape
        hidden_state = hidden_state.contiguous().view(batch_size,
                                                      self.hidden_channels)
        cell_state = cell_state.contiguous().view(batch_size,
                                                  self.hidden_channels)
        if x.shape[1] != hidden_state.shape[1]:
            x = self.input_proj(x)
        combined = torch.cat((x, hidden_state), -1)
        combined = self.layerNorm(combined)
        combined = activation_fn(combined)
        if self.nlayer_geatt3a > 0:
            attn_output = self.GEAttRecurrent_block1(
                Q=combined.unsqueeze(1),
                K=combined.unsqueeze(1),
                V=combined.unsqueeze(1),
            )
            combined = attn_output.squeeze(1)
        forget_gate = torch.sigmoid(self.forget_gate(combined))
        input_gate = torch.sigmoid(self.input_gate(combined))
        output_gate = torch.sigmoid(self.output_gate(combined))
        cell_candidate = torch.tanh(self.cell_gate(combined))
        if self.add_skips:
            gate_skip = torch.sigmoid(self.gate_skip(combined))
            next_cell = (forget_gate * cell_state + input_gate * cell_candidate
                         ) * (1 - gate_skip) + cell_state * gate_skip
        else:
            next_cell = forget_gate * cell_state + input_gate * cell_candidate
        next_cell = self.cell_norm(next_cell)
        next_hidden = output_gate * torch.tanh(next_cell)
        next_hidden = self.hidden_norm(next_hidden)
        next_hidden = activation_fn(next_hidden)
        if self.nlayer_geatt3b > 0:
            attn_output = self.GEAttRecurrent_block2(
                Q=next_hidden.unsqueeze(1),
                K=hidden_state.unsqueeze(1),
                V=hidden_state.unsqueeze(1),
            )
            next_hidden = attn_output.squeeze(1)
        if self.dropout_type == "zoneout":
            next_hidden = self.zoneout(
                prev_h=hidden_state,
                next_h=next_hidden,
                rate=self.dropout,
                training=self.training,
            )
            next_cell = self.zoneout(
                prev_h=cell_state,
                next_h=next_cell,
                rate=self.dropout,
                training=self.training,
            )
        return next_hidden, (next_hidden, next_cell)

    def zoneout(self, prev_h, next_h, rate, training=True):
        if training:
            d = torch.zeros_like(next_h,
                                 dtype=self.torch_dtype).bernoulli_(rate)
            next_h = d * prev_h + (1 - d) * next_h
        else:
            next_h = rate * prev_h + (1 - rate) * next_h
        return next_h

    def _get_activation(self, name):
        activation_functions = {
            "tanh": torch.tanh,
            "relu": F.relu,
            "selu": F.selu,
            "elu": F.elu,
            "gelu": F.gelu,
            "silu": nn.SiLU(),
            "softplus": F.softplus,
            "mish": lambda x: x * torch.tanh(F.softplus(x)),
            "sigmoid": torch.sigmoid,
            "identity": lambda x: x,
        }
        if name in activation_functions:
            return activation_functions[name]
        else:
            raise NotImplementedError(
                f"Activation function '{name}' is not implemented.")
