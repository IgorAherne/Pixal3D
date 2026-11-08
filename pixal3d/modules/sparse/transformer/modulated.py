from typing import *
import torch
import torch.nn as nn
from ..basic import VarLenTensor, SparseTensor
from ..attention import SparseMultiHeadAttention, SparseProjectAttention, SparseGatedProjectAttention
from ...norm import LayerNorm32
from ... import fused, int8mm, old_cards
from ..attention import SparseProjectAttention as _ProjectAttention
from .blocks import SparseFeedForwardNet


def _inner(h, dtype):
    """h in the sub-layers' dtype: a cast where the residual stream is wider than the torso (an fp16 torso keeps
    its stream in fp32, modules/utils.py stream_dtype: the late steps of the lr and hr flows reach 1.1e5 there,
    past fp16's range, while the sub-layers' own outputs stay below 5,000); h itself otherwise."""
    if dtype is None or getattr(h, "feats", h).dtype == dtype:
        return h
    return h.type(dtype)


def _stream(h, x):
    """h in the residual stream's dtype before its gate is applied (an fp16 h times a gate reaches past fp16's
    range where the stream is fp32); h itself where the two agree."""
    hx, xx = getattr(h, "feats", h), getattr(x, "feats", x)
    return h if hx.dtype == xx.dtype else h.type(xx.dtype)


def _modulate(h, scale, shift, dtype):
    """_inner(h * (1 + scale) + shift, dtype): one kernel on the cards below sm_80 (modules/old_cards.py: the same
    bits), the eager ops elsewhere."""
    out = old_cards.modulate(h, scale, shift, dtype)
    return out if out is not None else _inner(h * (1 + scale) + shift, dtype)


def _gated(x, h, gate):
    """x + _stream(h, x) * gate: one kernel on the cards below sm_80 (modules/old_cards.py: the same bits), the
    eager ops elsewhere."""
    out = old_cards.gated(x, h, gate)
    return out if out is not None else x + _stream(h, x) * gate


# A norm followed by an int8 layer of the Triton GEMMs writes that layer's int8 input itself (int8mm.norm_quant:
# norm_modulate's arithmetic and quantize_rows' in one kernel, the same bits at norm_modulate's warps) instead of a
# bf16 table the layer reads back to quantise. False: the bf16 hand-off everywhere (the same numbers).
NORM_QUANT = True


# The attentions' to_out (int8 layers of the Triton GEMMs), whose results go straight into the block's residual,
# write the new stream themselves: the residual kernel's arithmetic in their epilogue (int8mm.gemm `resid`; rounded
# mode, one prop), without the bf16 table of their results. False: the residual kernel (the same numbers).
RESID_EPILOGUE = True


def _pre_grain(layer: nn.Module, feats: torch.Tensor) -> Optional[int]:
    """The grain of the quantised input `layer` would make of `feats` (int8mm.GROUP, or 0: per row), or None where
    the norm hands it bf16 (NORM_QUANT off, or _int8_grain's None)."""
    return _int8_grain(layer, feats) if NORM_QUANT else None


def _resid(layer: nn.Module, feats: torch.Tensor, rows: dict, **operands) -> Optional[dict]:
    """The residual for `layer`'s epilogue ({"x": feats, **operands}), or None where the residual kernel serves:
    RESID_EPILOGUE off, another than the rounded mode, several props, or _int8_grain's None."""
    if (not RESID_EPILOGUE or fused.MODE != "rounded" or rows.get("batch") != 1 or "batch_index" in rows
            or _int8_grain(layer, feats) is None):
        return None
    gate = operands.get("gate")
    if gate is not None and gate.reshape(-1).shape[0] != feats.shape[-1]:
        return None
    return {"x": feats, **operands}


def _int8_grain(layer: nn.Module, feats: torch.Tensor) -> Optional[int]:
    """The grain of the quantised input `layer` would make of `feats` (int8mm.GROUP, or 0: per row), or None where
    `layer` is not an int8 layer of the Triton GEMMs, has a smoothed or shared input, or would not quantise them."""
    if layer is None:
        return None
    name = type(layer).__name__
    how = getattr(layer, "how", None)
    if name == "Int8Linear":
        if getattr(layer, "shared", None) is not None or getattr(layer, "inv_smooth", None) is not None:
            return None
    elif name == "Int8MLP":
        if (getattr(layer, "inv_smooth1", None) is not None or getattr(layer, "inv_smooth2", None) is not None
                or getattr(layer, "how2", None) != how):
            return None
    else:
        return None
    if how not in ("rows", "group"):
        return None
    grain = int8mm.GROUP if how == "group" else 0
    from ... import int8

    if (not int8._usable(feats, layer.in_features, layer.dtype) or not int8mm.ready(feats.device)[0]
            or int8mm.gemm_path() != "triton"):
        return None
    return grain


def _norm_quant(feats: torch.Tensor, eps: float, grain: int, rows: dict, **norm) -> tuple:
    """fused.norm_modulate(feats, eps, **norm, **rows) quantised for an int8 layer of that grain, in one kernel."""
    return int8mm.norm_quant(feats, eps, group=grain, rounded=fused.MODE != "fp32",
                             num_warps=8 if feats.shape[-1] > 1024 else 4, **norm, **rows)


def _int8_mlp(mlp: nn.Module, pre: tuple) -> torch.Tensor:
    """An int8 feed-forward (modules/int8.py Int8MLP) on its input quantised already: its own forward's GEMMs."""
    return int8mm.int8_mlp(pre, (mlp.w1_q, mlp.w1_s, mlp.b1), (mlp.w2_q, mlp.w2_s, mlp.b2),
                           path="group" if mlp.how == "group" else "rows", out_dtype=mlp.dtype)


def _self_half_shared(block, x: SparseTensor, run) -> SparseTensor:
    """`run(x)` (the block's norm1, self-attention and gated residual), or the stream the guided pair's first pass
    kept for the second. The guided sampler (pipelines/samplers/classifier_free_guidance_mixin.py) runs one x_t at
    one t twice, with the context and without; the first block's self-attention half reads no context, so its
    stream after it is the same bits in both passes. The sampler sets `cfg_pair` on the first block: {"keep": True}
    for the pair's first pass (the stream after the self-attention is kept in "x"), then {"keep": False, "x": ...}
    for the second, which starts from it."""
    pair = getattr(block, "cfg_pair", None)
    if pair is None:
        return run(x)
    if not pair.get("keep") and pair.get("x") is not None:
        feats = pair.pop("x")
        if feats.shape == x.feats.shape:
            return x.replace(feats)
    x = run(x)
    if pair.get("keep"):
        pair["x"] = x.feats
    return x


def _fused_rows(x: SparseTensor) -> Optional[dict]:
    """How the fused kernels find each row's batch item, or None when the block runs its eager code (the switch is
    off, or the features are not what the kernels take)."""
    feats = x.feats
    if not fused.active(feats) or feats.dim() != 2 or not feats.is_contiguous():
        return None
    batch = x.shape[0]
    if batch == 1:
        return {"batch": 1, "rows_per_batch": feats.shape[0]}
    return {"batch": batch, "batch_index": x.batch_boardcast_map}


class ModulatedSparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)

    def _forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        rows = _fused_rows(x)
        if rows is not None:
            h = x.replace(fused.norm_modulate(x.feats, self.norm1.eps, scale=scale_msa, shift=shift_msa, **rows))
            h = self.attn(h)
            x = x.replace(fused.residual(x.feats, h.feats, gate=gate_msa, **rows))
            h = x.replace(fused.norm_modulate(x.feats, self.norm2.eps, scale=scale_mlp, shift=shift_mlp, **rows))
            h = self.mlp(h)
            return x.replace(fused.residual(x.feats, h.feats, gate=gate_mlp, **rows))
        inner = getattr(self, "inner_dtype", None)
        h = x.replace(self.norm1(x.feats))
        h = _modulate(h, scale_msa, shift_msa, inner)
        h = self.attn(h)
        x = _gated(x, h, gate_msa)
        h = x.replace(self.norm2(x.feats))
        h = _modulate(h, scale_mlp, shift_mlp, inner)
        h = self.mlp(h)
        x = _gated(x, h, gate_mlp)
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    
    Supports two image attention modes:
    - "cross": Standard cross-attention with image features
    - "proj": Projection-based attention with view-aligned features
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        image_attn_mode: Literal["cross", "proj", "gated_proj"] = "cross",
        proj_in_channels: Optional[int] = None,
        vae_in_channels: Optional[int] = None,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.image_attn_mode = image_attn_mode
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        
        # Build cross attention based on mode
        if image_attn_mode == "cross":
            self.cross_attn = SparseMultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )
        elif image_attn_mode == "proj":
            _proj_in = proj_in_channels if proj_in_channels is not None else ctx_channels
            cross_attn_block = SparseMultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )
            self.cross_attn = SparseProjectAttention(cross_attn_block, channels, _proj_in)
        elif image_attn_mode == "gated_proj":
            _dino_in = proj_in_channels if proj_in_channels is not None else ctx_channels
            _vae_in = vae_in_channels if vae_in_channels is not None else 16
            cross_attn_block = SparseMultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )
            self.cross_attn = SparseGatedProjectAttention(cross_attn_block, channels, _dino_in, _vae_in)
        else:
            raise ValueError(f"Unknown image attention mode: {image_attn_mode}")
            
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)

    def _forward(self, x: SparseTensor, mod: torch.Tensor, context: Union[torch.Tensor, VarLenTensor],
                 zero: bool = False, proj_cache: bool = False) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        rows = _fused_rows(x)
        if rows is not None:
            return self._forward_fused(x, context, rows, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
                                       gate_mlp, zero, proj_cache)
        inner = getattr(self, "inner_dtype", None)

        def self_half(x):
            h = x.replace(self.norm1(x.feats))
            h = _modulate(h, scale_msa, shift_msa, inner)
            h = self.self_attn(h)
            return _gated(x, h, gate_msa)

        x = _self_half_shared(self, x, self_half)
        h = _inner(x.replace(self.norm2(x.feats)), inner)
        if type(self.cross_attn) is _ProjectAttention:
            # zero: the zero context's one row; proj_cache: proj_linear's kept answer (proj_attention.py)
            h = self.cross_attn(h, context, zero=zero, cache=proj_cache)
        else:
            h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = _modulate(h, scale_mlp, shift_mlp, inner)
        h = self.mlp(h)
        x = _gated(x, h, gate_mlp)
        return x

    def _forward_fused(self, x: SparseTensor, context: Union[torch.Tensor, VarLenTensor], rows: dict,
                       shift_msa: torch.Tensor, scale_msa: torch.Tensor, gate_msa: torch.Tensor,
                       shift_mlp: torch.Tensor, scale_mlp: torch.Tensor, gate_mlp: torch.Tensor,
                       zero: bool = False, proj_cache: bool = False) -> SparseTensor:
        """_forward with the elementwise ops fused (modules/fused.py: the switch PIXAL3D_FUSED); the same
        sub-layers in the same order."""
        def self_half(x):
            grain = _pre_grain(self.self_attn.to_qkv, x.feats)
            pre = (_norm_quant(x.feats, self.norm1.eps, grain, rows, scale=scale_msa, shift=shift_msa)
                   if grain is not None else None)
            h = x if pre is not None else x.replace(fused.norm_modulate(x.feats, self.norm1.eps, scale=scale_msa,
                                                                        shift=shift_msa, **rows))
            resid = _resid(self.self_attn.to_out, x.feats, rows, gate=gate_msa)
            h = self.self_attn(h, pre=pre, resid=resid)
            if resid is not None:
                return x.replace(h.feats)
            return x.replace(fused.residual(x.feats, h.feats, gate=gate_msa, **rows))

        x = _self_half_shared(self, x, self_half)
        tokens = x.feats.shape[0]
        cached = self.cross_attn.zero_row(tokens) if zero and type(self.cross_attn) is _ProjectAttention else None
        if cached is not None:
            # the zero context's one row for every token (proj_attention.py); norm2 is not needed for it
            x = x.replace(fused.residual_row(x.feats, cached))
        elif type(self.cross_attn) is _ProjectAttention:
            # SparseProjectAttention.forward, its sum (proj + global) done in the residual kernel (or in the cross
            # attention's to_out)
            global_context, proj_context = ((context['global'], context['proj']) if isinstance(context, dict)
                                            else context)
            block = self.cross_attn.cross_attn_block
            grain = _pre_grain(block.to_q, x.feats)
            pre = (_norm_quant(x.feats, self.norm2.eps, grain, rows, weight=self.norm2.weight, bias=self.norm2.bias)
                   if grain is not None else None)
            h = x if pre is not None else x.replace(fused.norm_modulate(x.feats, self.norm2.eps,
                                                                        weight=self.norm2.weight,
                                                                        bias=self.norm2.bias, **rows))
            resid, proj_feats = None, None
            if not zero and _resid(block.to_out, x.feats, rows) is not None:
                proj_feats = self.cross_attn.proj_feats(proj_context, proj_cache)
                if proj_feats.shape == x.feats.shape and proj_feats.is_contiguous():
                    resid = _resid(block.to_out, x.feats, rows, second=proj_feats)
            if resid is not None:
                x = x.replace(block(h, global_context, pre=pre, resid=resid).feats)
                del proj_feats, resid
            else:
                global_out = block(h, global_context, pre=pre)
                if proj_feats is None:
                    proj_feats = self.cross_attn.proj_feats(proj_context, proj_cache and not zero)
                if zero:
                    # the sum as the residual kernel rounds it (one bf16 add), first row
                    self.cross_attn.remember_zero_row(tokens, proj_feats[:1] + global_out.feats[:1])
                if proj_feats.shape == global_out.feats.shape and proj_feats.is_contiguous():
                    x = x.replace(fused.residual(x.feats, global_out.feats, second=proj_feats))
                else:
                    x = x.replace(fused.residual(x.feats, (proj_feats + global_out.feats).contiguous()))
                del global_out, proj_feats      # freed here, as on the eager path's return: not alive in the MLP
            del h, pre
        else:
            h = x.replace(fused.norm_modulate(x.feats, self.norm2.eps, weight=self.norm2.weight,
                                              bias=self.norm2.bias, **rows))
            h = self.cross_attn(h, context)
            x = x.replace(fused.residual(x.feats, h.feats))
        # (fc2 keeps the residual kernel: in its epilogue it measured no faster, the group GEMM's registers being
        # full)
        mlp = getattr(self.mlp, "mlp", None)
        grain = _pre_grain(mlp, x.feats)
        if grain is not None:
            h = x.replace(_int8_mlp(mlp, _norm_quant(x.feats, self.norm3.eps, grain, rows, scale=scale_mlp,
                                                     shift=shift_mlp)))
        else:
            h = x.replace(fused.norm_modulate(x.feats, self.norm3.eps, scale=scale_mlp, shift=shift_mlp, **rows))
            h = self.mlp(h)
        return x.replace(fused.residual(x.feats, h.feats, gate=gate_mlp, **rows))

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: Union[torch.Tensor, VarLenTensor],
                zero: bool = False, proj_cache: bool = False) -> SparseTensor:
        """`zero`: the context is all zeros (a guided sampling's unconditional pass); `proj_cache`: keep
        proj_linear's answer for the next pass with this context (proj_attention.py)."""
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, zero, proj_cache,
                                                     use_reentrant=False)
        else:
            return self._forward(x, mod, context, zero, proj_cache)
