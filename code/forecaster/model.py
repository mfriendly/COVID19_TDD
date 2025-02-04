import torch
import torch.nn as nn
import torch.nn.functional as F
from .Decoder import Decoder
from .Encoder import Encoder


class TriAttLSTM(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.torch_dtype = torch.float64 if args.dtype == "double" else torch.float32
        self.in_channels = args.in_channels
        self.hidden_channels = args.hidden_size
        self.output_size = args.output_size
        self.past_steps = args.past_steps
        self.future_steps = args.future_steps
        self.dropout = args.dropout
        self.device = args.device
        self.cl_decay_steps = args.cl_decay_steps = 300
        self.nlayer_geatt2 = args.nlayer_geatt2
        if self.nlayer_geatt2 > 0:
            from .GEAtt import GEAttention
            self.GEAttInputFeat_layers = nn.ModuleList([
                GEAttention(args, self.hidden_channels, 8, 0.6)
                for i in range(self.nlayer_geatt2)
            ])
            self.attention_layer_norms = nn.ModuleList([
                nn.LayerNorm(self.hidden_channels)
                for _ in range(self.nlayer_geatt2)
            ])
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.encoder_proj = nn.Linear(1, self.hidden_channels)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        feature_dim = self.hidden_channels
        self.final_proj = nn.Linear(feature_dim, self.hidden_channels)
        self.input_projection = None
        if self.in_channels != self.hidden_channels:
            self.input_projection = nn.Linear(self.in_channels,
                                              self.hidden_channels)
            self.input_projection_res = nn.Linear(self.in_channels,
                                                  self.hidden_channels)
        if self.nlayer_geatt2 == 0:
            self.input_projection = nn.Linear(self.in_channels,
                                              self.hidden_channels)
        #self.nlayer_skip2 = args.nlayer_skip2
        self.skip_norm = nn.LayerNorm(self.hidden_channels).to(
            self.device).double()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_in',
                                        nonlinearity='selu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_in',
                                        nonlinearity='selu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if 1:  #self.nlayer_skip2 > 0:
            self.skip_linear = nn.Linear(self.hidden_channels,
                                         self.hidden_channels)
            self.skip_norm = nn.LayerNorm(self.hidden_channels)
            if self.skip_norm is not None:
                init_weights(self.skip_norm)
        if self.input_projection is not None:
            self.input_projection.apply(init_weights)
        self.encoder_proj.apply(init_weights)
        self.final_proj.apply(init_weights)
        if 1:  #self.nlayer_skip2 >0  and self.skip_linear is not None:
            self.skip_linear.apply(init_weights)
        if self.input_projection is not None:
            self.layer_norm1 = nn.LayerNorm(self.hidden_channels)
        else:
            self.layer_norm2 = nn.LayerNorm(self.hidden_channels)
        self.layer_norm3 = nn.LayerNorm(1)
        if self.args.last_skip and self.args.last_linear:
            self.linear_out = nn.Linear(args.past_steps, args.future_steps)

    def forward(self, x, target_cl=None, task_level=42, global_step=None):
        batch_size, seq_length, _ = x.shape
        original_input = x
        print(f"==>> original_input.shape: {original_input.shape}")
        if 1:
            residual = original_input
        if self.nlayer_geatt2 > 0:
            for i in range(self.nlayer_geatt2):
                print("i", i)
                if i == 0:
                    x = self.input_projection(x)
                    print(f"==>> x.shape: {x.shape}")
                x = self.GEAttInputFeat_layers[i](
                    x, x, x) + self.input_projection_res(residual)
                if 1:
                    x = F.gelu(x)
                x = self.dropout_layer(x)
                x = self.attention_layer_norms[i](x)
                """residual was missing!"""
                x = self.dropout_layer(x)
                if i == 0 and self.input_projection is not None:
                    x = self.layer_norm1(x)
                    if self.args.last_skip:
                        x = x + original_input[:, :, 0:1]
                    x = F.gelu(x)
                else:
                    x = self.layer_norm2(x)
                    if self.args.last_skip:
                        x = x + residual[:, :, 0:1]
                    x = F.gelu(x)
        else:
            x = self.input_projection(x)
        encoder_outputs, _ = self.encoder(x, self.past_steps)
        if 1:
            encoder_outputs = encoder_outputs[:, :, 0:1]
        if 1:
            encoder_outputs = self.dropout_layer(
                self.encoder_proj(encoder_outputs))
        decoder_input = torch.zeros((batch_size, 1, 1),
                                    device=self.device,
                                    dtype=self.torch_dtype)
        outputs = self.decoder(decoder_input,
                               target_cl,
                               encoder_outputs,
                               task_level=task_level,
                               global_step=global_step)
        outputs = self.dropout_layer(outputs)
        if self.args.last_skip:
            if self.args.last_linear:
                orig_2 = self.linear_out(original_input[:, :, 0:1].permute(
                    0, 2, 1)).permute(0, 2, 1)
            else:
                orig_2 = original_input[:, :, 0:1]
            outputs = outputs + orig_2
        else:
            pass
        outputs = self.layer_norm3(outputs)
        return outputs
