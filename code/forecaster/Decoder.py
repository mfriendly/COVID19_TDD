import numpy as np
import torch
import torch.nn as nn
from .CustomLSTMCell import CustomLSTMCell


class Decoder(nn.Module):

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.torch_dtype = torch.float64 if args.dtype == "double" else torch.float32
        self.in_channels = 1
        self.hidden_channels = args.hidden_size
        self.output_size = args.output_size
        self.num_for_predict = args.num_for_predict
        self.dropout = args.dropout
        self.use_curriculum_learning = args.use_curriculum_learning
        self.cl_decay_steps = args.cl_decay_steps
        self.RNNCell = nn.ModuleList([CustomLSTMCell(args, self.in_channels)])
        from .GEAtt import GEAttention
        self.GEAttSeq2Seq_layers = nn.ModuleList([
            GEAttention(args, self.hidden_channels, 8, 0.6)
            for _ in range(args.nlayer_geatt1)
        ])
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(self.hidden_channels)
            for _ in range(args.nlayer_geatt1)
        ])
        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(self.hidden_channels)
            for _ in range(args.nlayer_geatt1)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels * 4),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_channels * 4, self.hidden_channels),
                nn.Dropout(self.dropout),
            ) for _ in range(args.nlayer_geatt1)
        ])
        self.context_linear = nn.Linear(self.hidden_channels * 2,
                                        self.hidden_channels)
        self.fc_final = nn.Linear(self.hidden_channels, self.output_size)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_channels)
            for _ in range(args.nlayer_geatt1)
        ])
        self.context_linear = nn.Linear(self.hidden_channels * 2,
                                        self.hidden_channels)
        self.fc_final = nn.Linear(self.hidden_channels, self.output_size)
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self,
                x,
                target_cl,
                encoder_outputs,
                task_level=42,
                global_step=None):
        x.shape[0]
        hidden_state = encoder_outputs[:, -1, :]
        cell_state = torch.zeros_like(hidden_state, dtype=self.torch_dtype)
        decoder_hidden = hidden_state.unsqueeze(1)
        outputs = []
        decoder_input = x
        for t in range(task_level):
            rnn_cell = self.RNNCell[0]
            rnn_out, (hidden_state,
                      cell_state) = rnn_cell(decoder_input,
                                             (hidden_state, cell_state))
            decoder_hidden = hidden_state.unsqueeze(1)
            attn_out = decoder_hidden
            for attention, norm1, norm2, ffn in zip(
                    self.GEAttSeq2Seq_layers,
                    self.layer_norms1,
                    self.layer_norms2,
                    self.ffn_layers,
            ):
                residual = attn_out
                feature = norm1(attn_out)
                attn_tmp = attention(Q=feature,
                                     K=encoder_outputs,
                                     V=encoder_outputs)
                if 1:
                    residual = attn_out
                    attn_out = residual + self.dropout_layer(ffn(attn_out))
            combined = torch.cat([decoder_hidden, attn_out], dim=-1)
            combined = self.context_linear(combined)
            combined = self.dropout_layer(combined)
            output = self.fc_final(combined)
            outputs.append(output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                prob = self._compute_sampling_threshold(global_step)
                if c < prob and t < target_cl.size(1) - 1:
                    decoder_input = target_cl[:, t:t + 1, :]
                else:
                    decoder_input = output
            else:
                decoder_input = output
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def _compute_sampling_threshold(self, global_step):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(global_step / self.cl_decay_steps))
