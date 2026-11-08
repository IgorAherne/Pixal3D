from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import VarLenTensor, SparseTensor
from .. import config
from ... import fused, old_cards
from .full_attn import sparse_scaled_dot_product_attention
from .windowed_attn import sparse_windowed_scaled_dot_product_self_attention
from .rope import SparseRotaryPositionEmbedder


def _feats(x: Union[VarLenTensor, torch.Tensor]) -> torch.Tensor:
    return x.feats if isinstance(x, VarLenTensor) else x


def _pre_linear(layer: nn.Module, pre: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """An int8 layer (modules/int8.py Int8Linear) on its input quantised already (int8mm.norm_quant's rows and
    scales): the GEMM its own forward runs after quantising a bf16 input, [M, N] in the layer's dtype."""
    from ... import int8mm

    return int8mm.int8_linear(pre, layer.weight_q, layer.scale, layer.bias,
                              path="triton" if layer.how == "rows" else "group", out_dtype=layer.dtype)


def _resid_linear(layer: nn.Module, feats: torch.Tensor, resid: dict) -> torch.Tensor:
    """An int8 layer (Int8Linear) on its bf16 input with the block's residual in its epilogue (int8mm.gemm `resid`):
    the new stream, [M, N]."""
    from ... import int8mm

    return int8mm.int8_linear(feats.reshape(-1, feats.shape[-1]), layer.weight_q, layer.scale, layer.bias,
                              path="triton" if layer.how == "rows" else "group", out_dtype=layer.dtype, resid=resid)


class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: Union[VarLenTensor, torch.Tensor]) -> Union[VarLenTensor, torch.Tensor]:
        x_type = x.dtype
        x = x.float()
        if isinstance(x, VarLenTensor):
            x = x.replace(F.normalize(x.feats, dim=-1) * self.gamma * self.scale)
        else:
            x = F.normalize(x, dim=-1) * self.gamma * self.scale
        return x.to(x_type)


class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed", "double_windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        rope_freq: Tuple[int, int] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed", "double_windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        assert type == "self" or use_rope is False, "Rotary position embeddings only supported for self-attention"
        if attn_mode == 'double_windowed':
            assert window_size % 2 == 0, "Window size must be even for double windowed attention"
            assert num_heads % 2 == 0, "Number of heads must be even for double windowed attention"
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        
        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            
        self.to_out = nn.Linear(channels, channels)
        self.sage_folded = False        # sage.fold: v's 2^-8 and the answer's 2^8 live in to_qkv / to_out
        self.sage_int8pv = False        # sage.mark_int8pv: this block's self-attention may run the int8 P.V kernel

        if use_rope:
            self.rope = SparseRotaryPositionEmbedder(self.head_dim, rope_freq=rope_freq)

    @staticmethod
    def _linear(module: nn.Linear, x: Union[VarLenTensor, torch.Tensor]) -> Union[VarLenTensor, torch.Tensor]:
        if isinstance(x, VarLenTensor):
            return x.replace(module(x.feats))
        else:
            return module(x)

    @staticmethod
    def _reshape_chs(x: Union[VarLenTensor, torch.Tensor], shape: Tuple[int, ...]) -> Union[VarLenTensor, torch.Tensor]:
        if isinstance(x, VarLenTensor):
            return x.reshape(*shape)
        else:
            return x.reshape(*x.shape[:2], *shape)

    def _fused_pre(self, x: Union[VarLenTensor, torch.Tensor], num_fused: int) -> Union[VarLenTensor, torch.Tensor]:
        if isinstance(x, VarLenTensor):
            x_feats = x.feats.unsqueeze(0)
        else:
            x_feats = x
        x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
        return x.replace(x_feats.squeeze(0)) if isinstance(x, VarLenTensor) else x_feats
    
    def _fused_qk(self, q: Union[VarLenTensor, torch.Tensor], k: Union[VarLenTensor, torch.Tensor],
                  rope: bool = True) -> Optional[Tuple[Union[VarLenTensor, torch.Tensor], ...]]:
        """q and k through the RMS norm (and the rotary embedding when the block uses it) in one fused kernel each
        (modules/fused.py), or None when their layout does not suit it: the eager code runs then."""
        phases = self.rope.phases(q) if rope and self.use_rope else None
        fq = fused.qk_norm_rope(_feats(q), self.q_rms_norm.gamma, self.q_rms_norm.scale, phases)
        if fq is None:
            return None
        fk = fused.qk_norm_rope(_feats(k), self.k_rms_norm.gamma, self.k_rms_norm.scale, phases)
        if fk is None:
            return None
        wrap = lambda t, f: t.replace(f) if isinstance(t, VarLenTensor) else f
        return wrap(q, fq), wrap(k, fk)

    def forward(self, x: SparseTensor, context: Optional[Union[VarLenTensor, torch.Tensor]] = None,
                pre: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, resid: Optional[dict] = None) -> SparseTensor:
        """`pre`: the input of to_qkv (self) or to_q (cross) quantised already for its int8 layer (the block's
        norm_quant); x then gives only the layout. `resid`: the block's residual done in to_out's epilogue (int8mm.gemm
        `resid`): the result is then the new stream."""
        if self._type == "self":
            qkv = self._linear(self.to_qkv, x) if pre is None else x.replace(_pre_linear(self.to_qkv, pre))
            qkv = self._fused_pre(qkv, num_fused=3)
            apart = None
            if self.qk_rms_norm or self.use_rope:
                q, k, v = qkv.unbind(dim=-3)
                normed = self._fused_qk(q, k) if self.qk_rms_norm and fused.active(q.feats) else None
                if normed is None and self.qk_rms_norm:     # the eager chain's bits in one kernel each
                    normed = old_cards.qk(self, q, k, rope=True)
                if normed is not None:
                    q, k = normed
                else:
                    if self.qk_rms_norm:
                        q = self.q_rms_norm(q)
                        k = self.k_rms_norm(k)
                    if self.use_rope:
                        q, k = self.rope(q, k)
                if self.attn_mode == "full" and config.ATTN in ('flash_attn', 'sdpa'):
                    # flash_attn's packed call only slices q, k and v back out of the stack and runs the kernel
                    # of the unpacked call with the same arguments, so the stack would be a copy for nothing;
                    # torch's SDPA takes them apart as well. The call with the three is also the only one that
                    # offers one sequence's self-attention to SageAttention (full_attn.py one_self).
                    apart = (q, k, v)
                else:
                    qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
            if self.attn_mode == "full":
                h = (sparse_scaled_dot_product_attention(*apart, v_scaled=self.sage_folded, int8pv=self.sage_int8pv)
                     if apart
                     else sparse_scaled_dot_product_attention(qkv))
            elif self.attn_mode == "windowed":
                h = sparse_windowed_scaled_dot_product_self_attention(
                    qkv, self.window_size, shift_window=self.shift_window
                )
            elif self.attn_mode == "double_windowed":
                qkv0 = qkv.replace(qkv.feats[:, :, self.num_heads//2:])
                qkv1 = qkv.replace(qkv.feats[:, :, :self.num_heads//2])
                h0 = sparse_windowed_scaled_dot_product_self_attention(
                    qkv0, self.window_size, shift_window=(0, 0, 0)
                )
                h1 = sparse_windowed_scaled_dot_product_self_attention(
                    qkv1, self.window_size, shift_window=tuple([self.window_size//2] * 3)
                )
                h = qkv.replace(torch.cat([h0.feats, h1.feats], dim=1))
        else:
            q = self._linear(self.to_q, x) if pre is None else x.replace(_pre_linear(self.to_q, pre))
            q = self._reshape_chs(q, (self.num_heads, -1))
            kv = self._linear(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)
            if self.qk_rms_norm:
                k, v = kv.unbind(dim=-3)
                normed = self._fused_qk(q, k, rope=False) if fused.active(_feats(q)) else None
                if normed is None:
                    normed = old_cards.qk(self, q, k, rope=False)
                if normed is not None:
                    q, k = normed
                else:
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                h = sparse_scaled_dot_product_attention(q, k, v)
            else:
                h = sparse_scaled_dot_product_attention(q, kv)
        h = self._reshape_chs(h, (-1,))
        if resid is not None:
            return h.replace(_resid_linear(self.to_out, _feats(h), resid))
        h = self._linear(self.to_out, h)
        return h
