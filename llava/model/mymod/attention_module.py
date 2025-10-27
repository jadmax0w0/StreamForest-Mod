import sys
sys.path.extend(['.', './model'])
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
import math
from typing import Optional, Union, Tuple
try:
    from .mm_utils import OutputScaling, printr
except ImportError:
    from model.modular.mm_utils import OutputScaling, printr


class MultiQueryAttention(nn.Module):
    def __init__(self, d_model=2048, num_heads=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        
        # Multi-Query优化：共享KV投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.head_dim)  # 共享KV头维度
        self.v_proj = nn.Linear(d_model, self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

        self.resid_dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # 投影
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # 重排维度
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        
        # 高效注意力计算
        attn_weights = torch.einsum("bhid, bjd -> bhij", q, k) / (self.head_dim ** 0.5)
        attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), -1e9)
        attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values  # 数值稳定 softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 注意力聚合
        output = torch.einsum("bhij, bjd -> bhid", attn_weights, v)
        output = rearrange(output, "b h s d -> b s (h d)")
        
        return self.resid_dropout(self.out_proj(output))


class MultiheadAttentionObsolete(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_k, x_v, attn_mask=None):
        B, L, _ = x_q.shape

        def reshape(x):
            return x.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        q = reshape(self.q_proj(x_q))
        k = reshape(self.k_proj(x_k))
        v = reshape(self.v_proj(x_v))

        if attn_mask is not None:
            assert attn_mask.dtype in [torch.bool, torch.float32], "attn_mask must be bool or float32"

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.embed_dim)
        return self.out_proj(self.attn_dropout(attn_out))


class MultiheadAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            q_dim: int,
            k_dim: Optional[int] = None,
            v_dim: Optional[int] = None,
            out_dim: Optional[int] = None,
            use_q_proj: bool = True,
            use_k_proj: bool = True,
            use_v_proj: bool = True,
            use_out_proj: bool = True,
            dropout: float = 0.1,
            **kwargs
    ):
        super().__init__(**kwargs)
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        if k_dim is None:
            k_dim = q_dim
        if v_dim is None:
            v_dim = q_dim
        if out_dim is None:
            out_dim = q_dim

        self.q_proj = nn.Linear(q_dim, embed_dim) if use_q_proj else nn.Identity()
        self.k_proj = nn.Linear(k_dim, embed_dim) if use_k_proj else nn.Identity()
        self.v_proj = nn.Linear(v_dim, embed_dim) if use_v_proj else nn.Identity()
        self.out_proj = nn.Linear(embed_dim, out_dim) if use_out_proj else nn.Identity()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.use_q_proj = use_q_proj
        self.use_k_proj = use_k_proj
        self.use_v_proj = use_v_proj
        self.use_out_proj = use_out_proj

        self.attn_dropout = nn.Dropout(dropout)
    
    def weight_init(self):
        init_scale = 0.02
        small_scale = 0.01
        
        if self.use_q_proj:
            nn.init.xavier_normal_(self.q_proj.weight, gain=init_scale)
            if self.q_proj.bias is not None:
                nn.init.zeros_(self.q_proj.bias)
        
        if self.use_k_proj:
            nn.init.normal_(self.k_proj.weight, mean=0.0, std=init_scale * 0.7)
            if self.k_proj.bias is not None:
                nn.init.zeros_(self.k_proj.bias)
        
        if self.use_v_proj:
            nn.init.normal_(self.v_proj.weight, mean=0.0, std=init_scale * 0.7)
            if self.v_proj.bias is not None:
                nn.init.zeros_(self.v_proj.bias)
        
        if self.use_out_proj:
            nn.init.xavier_normal_(self.out_proj.weight, gain=small_scale)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, q, k, v, attn_mask = None, **kwargs):
        """
        Args:
            attn_mask: `False` means masked and their attn scores will be ignored, `True` means unmasked
        Returns:
            (attn_out, attn_score): shaped like `[B, L, D]`, `[B, H, L, S]`
        """
        B, L, D = q.shape
        _, S, _ = k.shape

        def reshape(x):
            b, l, _ = x.shape
            return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        q, k, v = reshape(q), reshape(k), reshape(v)

        scale_factor = 1.0 / math.sqrt(self.head_dim)
        attn_bias = torch.zeros((L, S), dtype=q.dtype, device=q.device)
        if attn_mask is not None:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        
        attn_score = q @ k.transpose(-1, -2) * scale_factor  # [B, H, L, d] @ [B, H, d, S] = [B, H, L, S]
        attn_score += attn_bias
        attn_score = torch.softmax(attn_score, dim=-1)
        # When an entire row is masked, the elements' attn scores will be nan
        attn_score = attn_score.masked_fill(attn_score.isnan(), 0.0)
        attn_score = self.attn_dropout(attn_score)

        attn_out = attn_score @ v  # [B, H, L, d]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
        attn_out = self.out_proj(attn_out)

        return attn_out, attn_score


class ThresholdMultiheadAttention(MultiheadAttention):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            q_dim: int,
            k_dim: Optional[int] = None,
            v_dim: Optional[int] = None,
            out_dim: Optional[int] = None,
            use_q_proj: bool = True,
            use_k_proj: bool = True,
            use_v_proj: bool = True,
            use_out_proj: bool = True,
            dropout: float = 0.1,
            **kwargs
    ):
        super().__init__(embed_dim, num_heads, q_dim, k_dim, v_dim, out_dim, use_q_proj, use_k_proj, use_v_proj, use_out_proj, dropout, **kwargs)
    
    def weight_init(self):
        return super().weight_init()

    def forward(self, q, k, v, attn_mask = None, score_threshold = float("-inf"), keep_diagonal_scores = False):
        """
        Args:
            attn_mask: shape `[L, S]`; `False` means masked and their attn scores will be ignored, `True` means unmasked
            score_threshold: mask the elements in attn_score when their values are less than this threshold
        Returns:
            (attn_out, attn_score): shaped like `[B, L, D]`, `[B, H, L, S]`
        """
        B, L, D = q.shape
        _, S, _ = k.shape
        H = self.num_heads

        def reshape(x):
            b, l, _ = x.shape
            return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        q, k, v = reshape(q), reshape(k), reshape(v)

        ## Attention mask ##
        attn_bias = torch.zeros((L, S), dtype=q.dtype, device=q.device)
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)  # [B, H, L, S]
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1) if attn_mask is not None else None
        
        # Apply attn score thresholding (based on cos sims between all Q, K vector pairs)
        q_normed = F.normalize(q, p=2, dim=-1)  # [B, H, L, d]
        k_normed = F.normalize(k, p=2, dim=-1)  # [B, H, S, d]
        cos_sims = q_normed @ k_normed.transpose(-1, -2)
        threshold_mask = cos_sims < score_threshold  # 每个头都有一个独立的 threshold mask
        if keep_diagonal_scores and threshold_mask.shape[-1] == threshold_mask.shape[-2]:
            # When self-attn, make sure diagonal scores are not affected by threshold mask (a vector should always be related to itself)
            n = threshold_mask.shape[-1]
            threshold_mask.masked_fill_(torch.eye(n, dtype=torch.bool, device=threshold_mask.device), False)
        
        # Apply both the input attn mask and the threshold mask
        attn_bias = attn_bias.masked_fill(threshold_mask, float("-inf"))
        if attn_mask is not None:
            attn_bias = attn_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
        
        ## Attention calculation ##
        scale_factor = 1.0 / math.sqrt(self.head_dim)
        attn_score = q @ k.transpose(-1, -2) * scale_factor  # [B, H, L, d] @ [B, H, d, S] = [B, H, L, S]

        attn_score += attn_bias
        attn_score = torch.softmax(attn_score, dim=-1)
        # When an entire row is masked, the elements' attn scores will be nan
        attn_score = attn_score.masked_fill(attn_score.isnan(), 0.0)
        attn_score = self.attn_dropout(attn_score)

        attn_out = attn_score @ v  # [B, H, L, d]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
        attn_out = self.out_proj(attn_out)

        return attn_out, attn_score


class GraphLikeMultiheadAttention(MultiheadAttention):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            q_dim: int,
            k_dim: Optional[int] = None,
            v_dim: Optional[int] = None,
            out_dim: Optional[int] = None,
            use_q_proj: bool = True,
            use_k_proj: bool = True,
            use_v_proj: bool = True,
            use_out_proj: bool = True,
            dropout: float = 0.1,
            **kwargs
    ):
        super().__init__(embed_dim, num_heads, q_dim, k_dim, v_dim, out_dim, use_q_proj, use_k_proj, use_v_proj, use_out_proj, dropout, **kwargs)
    
    def weight_init(self):
        return super().weight_init()
    
    def forward(self, q, k, v, attn_mask = None, masked_bool_value = False, adjacency_mat: torch.BoolTensor = None, connectivity_mat: Tensor = None, **kwargs):
        """
        Args:
            adjacency_mat (BoolTensor): shaped `[B, L, S]` or `[B, H, L, S]`. a pure 0/1 matrix to represent the graph. no gradient is needed on this matrix
            connectivity_mat (Tensor): shaped `[B, L, S]` or `[B, H, L, S]`. if present, this will get multiplied with attention logits
                (then masked and softmax-ed to get attn-score)
        """
        B, L, D = q.shape
        _, S, _ = k.shape
        H = self.num_heads

        Q = q.clone()

        def reshape(x):
            b, l, _ = x.shape
            return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        ## Attention mask ##
        attn_bias = torch.zeros((B, H, L, S), dtype=q.dtype, device=q.device)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1) if attn_mask.dim() == 2 else attn_mask  # [L, S] -> [B, H, L, S]
            attn_bias.masked_fill_(attn_mask.logical_not() if masked_bool_value == False else attn_mask, float("-inf"))
        if adjacency_mat is not None:
            # Expand adjacency mat from [B, L, S] to [B, H, L, S] if needed
            if adjacency_mat.dim() == 3:
                adjacency_mat = adjacency_mat.unsqueeze(1).expand(-1, H, -1, -1)
            attn_bias.masked_fill_(adjacency_mat.logical_not(), float("-inf"));

        ## Attention calculations ##
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        q, k, v = reshape(q), reshape(k), reshape(v)
        
        scale_factor = 1.0 / math.sqrt(self.head_dim)
        attn_score = q @ k.transpose(-1, -2) * scale_factor  # [B, H, L, d] @ [B, H, d, S] = [B, H, L, S]

        # Apply connectivity matrix to attn score
        if connectivity_mat is not None:
            # Expand connectivity mat from [B, L, S] to [B, H, L, S] if needed
            if connectivity_mat.dim() == 3:
                connectivity_mat = connectivity_mat.unsqueeze(1).expand(-1, H, -1, -1)
            # attn_score *= connectivity_mat
            eps = 1e-6
            attn_score += 2 * torch.log(connectivity_mat + eps)
        
        attn_score += attn_bias

        # Pre-eliminate all rows with only inf's to avoid nan's
        score_infmask = torch.isinf(attn_score).all(dim=-1, keepdim=True).expand(-1, -1, -1, S)
        attn_score = attn_score.masked_fill(score_infmask, 0.0)
        attn_score = torch.softmax(attn_score, dim=-1)
        attn_score = attn_score.masked_fill(score_infmask, 0.0)

        attn_score = self.attn_dropout(attn_score)

        attn_out = attn_score @ v  # [B, H, L, d]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
        attn_out = self.out_proj(attn_out)

        return attn_out, attn_score


class AttentionBlock(nn.Module):
    def __init__(
            self,
            d_model = 2048,
            num_heads = 16,
            ff_dim = 4096,
            attn_dropout = 0.1,
            ffn_dropout = 0.1,
            use_qkvo_proj = (True, True, True, True),
            query_residual_scale = None,
            value_residual_scale = None,
            ffn_residual_scale = None,
            attn_implementation = MultiheadAttention,
            k_residual = False,  # used in self-attn
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.k_residual = k_residual
        self.use_qres = query_residual_scale is not None and query_residual_scale > 0
        self.use_vres = value_residual_scale is not None and value_residual_scale > 0

        if attn_implementation is None:
            attn_implementation = MultiheadAttention
        
        self.attn = attn_implementation(
            embed_dim=d_model,
            num_heads=num_heads,
            q_dim=d_model,
            use_q_proj=use_qkvo_proj[0],
            use_k_proj=use_qkvo_proj[1],
            use_v_proj=use_qkvo_proj[2],
            use_out_proj=use_qkvo_proj[3],
            dropout=attn_dropout,
        )
        self.attn_norm = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(ffn_dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        self.ffn_scale = nn.Identity()
        if self.use_qres or self.use_vres:
            ffn_residual_scale = ffn_residual_scale if ffn_residual_scale is not None else 1.0
            self.ffn_scale = OutputScaling(ffn_residual_scale, dim=d_model)
        if self.use_qres:
            self.query_scale = OutputScaling(query_residual_scale, dim=d_model)
        if self.use_vres:
            self.value_scale = OutputScaling(value_residual_scale, dim=d_model)
    
    def weight_init(self):
        for module in (self.attn_norm, self.ffn_norm):
            nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, q, k, v, qres, vres, attn_mask = None, **kwargs):
        attn_out, attn_score = self.attn(q, k, v, attn_mask, **kwargs)

        if not self.k_residual:
            attn_adnorm = self.attn_norm(q + attn_out)
        else:
            attn_adnorm = self.attn_norm(q + F.interpolate(k.transpose(-1, -2), q.shape[1], mode='linear').transpose(-1, -2) + attn_out)
        
        ffn_out     = self.ffn(attn_adnorm)
        ffn_adnorm  = self.ffn_norm(attn_adnorm + ffn_out)

        # Residual connections
        qres2 = self.query_scale(qres) if self.use_qres else torch.zeros_like(qres)
        if self.use_vres:
            # 所有 head 的 attn score 先求平均
            attn_score  = attn_score.mean(dim=1)  # [B, H, L, S] -> [B, L, S]
            vres2       = self.value_scale(attn_score @ vres)
        else:
            vres2       = torch.zeros_like(qres)

        return qres2 + vres2 + self.ffn_scale(ffn_adnorm)


class MemoryAttentionDecoder(nn.Module):
    def __init__(
            self,
            num_layers = 1,
            d_model = 2048,
            num_heads = 16,
            ff_dim = 4096,
            attn_dropout = 0.1,
            ffn_dropout = 0.1,
            use_qkvo_proj: Tuple[bool] = (True, True, True, True),
            only_residual_last_layer: bool = True,
            query_residual_scale: Optional[float] = None,
            value_residual_scale: Optional[float] = None,
            ffn_residual_scale: Optional[float] = None,
            attn_implementation: Optional[type] = None,
            k_residual: Optional[bool] = None,
    ):
        super().__init__()
        if only_residual_last_layer:
            self.layers = nn.ModuleList([
                AttentionBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                    use_qkvo_proj=use_qkvo_proj,
                    query_residual_scale=query_residual_scale if i == num_layers - 1 else None,
                    value_residual_scale=value_residual_scale if i == num_layers - 1 else None,
                    ffn_residual_scale=ffn_residual_scale if i == num_layers - 1 else None,
                    attn_implementation=attn_implementation,
                    k_residual=k_residual,
                )
                for i in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                AttentionBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                    use_qkvo_proj=use_qkvo_proj,
                    query_residual_scale=query_residual_scale,
                    value_residual_scale=value_residual_scale,
                    ffn_residual_scale=ffn_residual_scale,
                    attn_implementation=attn_implementation,
                    k_residual=k_residual,
                )
                for _ in range(num_layers)
            ])
    
    def forward(
            self,
            query: Tensor,
            memory: Tensor,
            query_residual: Optional[Tensor] = None,
            value_residual: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
            **kwargs,
    ):
        """
        Args:
            query_residual: 为 None 则默认取输入的 query
            value_residual: 为 None 则默认取输入的 value
        """
        h = query
        query_residual = query if query_residual is None else query_residual
        value_residual = memory if value_residual is None else value_residual
        for layer in self.layers:
            h = layer(h, memory, memory, qres=query_residual, vres=value_residual, attn_mask=attn_mask, **kwargs)
        return h


@dataclass
class IntraFrameDecoderConfig:
    num_layers: int = 2
    d_model: int = 3584
    num_heads: int = 16
    ff_dim: int = 4096
    attn_dropout: float = 0.1
    ffn_dropout: float = 0.1
    use_qkvo_proj: tuple[bool] = (True, True, True, True)
    only_residual_last_layer: bool = True
    query_residual_scale: Optional[float] = None
    value_residual_scale: Optional[float] = None
    ffn_residual_scale: Optional[float] = None


@dataclass
class InterFrameDecoderConfig:
    num_queries: int = 64
    num_layers: int = 3
    d_model: int = 3584
    num_heads: int = 16
    ff_dim: int = 4096
    attn_dropout: float = 0.1
    ffn_dropout: float = 0.1
    use_qkvo_proj: tuple[bool] = (True, True, True, True)
    only_residual_last_layer: bool = True
    query_residual_scale: Optional[float] = None
    value_residual_scale: Optional[float] = 1.0
    ffn_residual_scale: Optional[float] = 0.5


@dataclass
class InterClipDecoderConfig:
    num_layers: int = 2
    d_model: int = 3584
    num_heads: int = 16
    ff_dim: int = 4096
    attn_dropout: float = 0.1
    ffn_dropout: float = 0.1
    use_qkvo_proj: tuple[bool] = (True, True, True, True)
    only_residual_last_layer: bool = True
    query_residual_scale: Optional[float] = 0.9  # 90% 的原 query 信息，10% 的新 attended 信息
    value_residual_scale: Optional[float] = None
    ffn_residual_scale: Optional[float] = 0.1


if __name__ == "__main__":
    d1 = MemoryAttentionDecoder(
        num_layers=2, d_model=16, num_heads=4, ff_dim=20,
    )
    d2 = MemoryAttentionDecoder(
        num_layers=2, d_model=16, num_heads=4, ff_dim=20,
        query_residual_scale=0.5,
    )
    d3 = MemoryAttentionDecoder(
        num_layers=2, d_model=16, num_heads=4, ff_dim=20,
        value_residual_scale=0.6, ffn_residual_scale=0.9
    )
    d4 = MemoryAttentionDecoder(
        num_layers=2, d_model=16, num_heads=4, ff_dim=20,
        query_residual_scale=0.4, value_residual_scale=0.4, only_residual_last_layer=False
    )
    d5 = MemoryAttentionDecoder(
        num_layers=2, d_model=16, num_heads=4, ff_dim=20,
        use_qkvo_proj=(False, False, False, False),
        query_residual_scale=0.4, ffn_residual_scale=0.8,
    )

    a = torch.randn(2, 10, 16)
    b = torch.randn(2, 12, 16)
    ar = torch.randn(2, 10, 16)
    br = torch.randn(2, 12, 16)

    c1 = d1.forward(a, b)
    c2 = d2.forward(a, b, ar, br)
    c3 = d3.forward(a, b)
    c4 = d4.forward(a, b, ar, br)
    c5 = d5.forward(a, b)
    pass