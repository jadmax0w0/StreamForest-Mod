## Adopted from MS-TCN++ ##
## [Github] https://github.com/sj-li/MS-TCN2 ##
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, hidden_dim, in_dim, out_dim, dropout: float = 0.05):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, hidden_dim, in_dim, out_dim, dropout=dropout)
        self.Rs = nn.ModuleList([Refinement(num_layers_R, hidden_dim, out_dim, out_dim) for _ in range(num_R)])
    
    def weight_init(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x (Tensor): `[B, C_in, N]`
        Returns:
            Tensor: `[B, C_out, N]`
        """
        out = self.PG(x)
        for rid, R in enumerate(self.Rs):
            # ❗ Time insentive
            out = R(out)  # removed softmax function, since no 'class scores' is needed in context of visual understanding
        return out


class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, hidden_dim, in_dim, out_dim, norm_groups: int = 4, dropout: float = 0.05):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(in_dim, hidden_dim, 1)

        self.conv_dilated_1 = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ])

        self.conv_dilated_2 = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=2**i, dilation=2**i)
            for i in range(num_layers)
        ])

        self.conv_fusion = nn.ModuleList([
            nn.Conv1d(2*hidden_dim, hidden_dim, 1) for i in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.drop_norm = nn.GroupNorm(norm_groups, hidden_dim)
        self.conv_out = nn.Conv1d(hidden_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: `[B, in_dim, N]`
        Returns:
            `[B, out_dim, N]`
        """
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))  # ❗ Time insentive
            f = F.gelu(f)
            f = self.dropout(f)
            f = f_in + self.drop_norm(f)

        out = self.conv_out(f)

        return out


class Refinement(nn.Module):
    def __init__(self, num_layers, hidden_dim, in_dim, out_dim, norm_groups: int = 4, dropout: float = 0.05):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.layers = nn.ModuleList([DilatedResidualLayer(2**i, hidden_dim, hidden_dim, dropout=dropout)
                                     for i in range(num_layers)])
        self.conv_out = nn.Conv1d(hidden_dim, out_dim, 1)
        self.out_norm = nn.GroupNorm(norm_groups, out_dim)

    def forward(self, x):
        """
        Args:
            x: `[B, in_dim, N]`
        Returns:
            `[B, out_dim, N]`
        """
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.out_norm(self.conv_out(out))

        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, norm_groups: int = 4, dropout: float = 0.05):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.drop_norm = nn.GroupNorm(norm_groups, out_channels)

    def forward(self, x):
        out = F.gelu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + self.drop_norm(out)