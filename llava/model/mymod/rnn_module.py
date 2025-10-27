import sys
sys.path.extend([".", "./model"])
import torch
from torch import nn, Tensor
try:
    from .attention_module import MultiheadAttention
except ImportError:
    from model.modular.attention_module import MultiheadAttention


class GroupGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, **kwargs):
        super().__init__(**kwargs)

        self.w_i = nn.Linear(input_dim, 3 * hidden_dim, bias=True)
        self.w_h = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.seqlen_adaptor = MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,  # FIXME: hard-coded for now
            q_dim=input_dim,
            k_dim=hidden_dim,
            use_q_proj=True,    # 1. 将输入的 query, key 放到同一模态中
            use_k_proj=True,
            use_v_proj=False,   # 2. 保持 output 的模态与输入的 value 相同
            use_out_proj=False,
        )
    
    def weight_init(self):
        for module in (self.w_i, self.w_h):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: Tensor, h_prev: Tensor):
        """
        将输入的一个序列视为一个单体
        Args:
            x: shape `[B, L, D]`
            h_prev: shape `[B, L_prev, D]`
        Returns:
            Tensor: `h_t` shaped `[B, L, D]`
        """
        # Seqlen adapt
        # -> Shaped like [B, L, D]
        h, _ = self.seqlen_adaptor(q=x, k=h_prev, v=h_prev)

        # -> Shaped like [B, L, 3D]
        xs = self.w_i(x)
        hs = self.w_h(h)

        # 切分为 r, z, n (reset, update, new)
        # -> Shaped like [B, L, D]
        x_r, x_z, x_n = torch.chunk(xs, 3, dim=-1)
        h_r, h_z, h_n = torch.chunk(hs, 3, dim=-1)

        reset_gate = torch.sigmoid(x_r + h_r)
        update_gate = torch.sigmoid(x_z + h_z)

        h_canddt = torch.tanh(x_n + reset_gate * h_n)
        h_t = (1.0 - update_gate) * h + update_gate * h_canddt

        return h_t


class GroupGRU(nn.Module):
    def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, **kwargs):
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.hidden_states = [None for _ in range(num_layers)]

        self.gru_cells = nn.ModuleList([
            GroupGRUCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.gru_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim, bias=True),
        )
        self.out_norm = nn.LayerNorm(input_dim)
    
    def weight_init(self):
        for module in self.gru_norms:
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.out_proj:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.ones_(self.out_norm.weight)
        if hasattr(self.out_norm, 'bias') and self.out_norm.bias is not None:
            nn.init.zeros_(self.out_norm.bias)
    
    def forward(self, x: Tensor, detach_h_prev: bool = False):
        """
        Args:
            x: shaped `[B, L, D=input_dim]`
        """
        x_next = x
        for i, layer in enumerate(self.gru_cells):
            h_prev = self.hidden_states[i]
            if h_prev is None:
                B, L, _ = x_next.shape
                h_prev = torch.zeros(B, L, self.hidden_dim, dtype=x_next.dtype, device=x_next.device, requires_grad=x_next.requires_grad)
            if detach_h_prev:
                h_prev = h_prev.detach()
            
            x_next = layer(x_next, h_prev)
            self.hidden_states[i] = x_next.clone()
            x_next = self.gru_norms[i](x_next)
        
        y = self.out_norm(self.out_proj(x_next))
        return y
    
    def clear_states(self):
        self.hidden_states = [None for _ in range(self.num_layers)]


if __name__ == "__main__":
    # gru = GroupGRUCell(10, 20, True)
    gru = GroupGRU(3, 10, 20)

    x = torch.randn(2, 5, 10)
    # h0 = torch.randn(2, 3, 20)

    # h1, y = gru(x, h0)
    y = gru.forward(x, detach_h_prev=True)

    x = torch.randn(2, 3, 10)
    y = gru.forward(x, detach_h_prev=True)
    pass