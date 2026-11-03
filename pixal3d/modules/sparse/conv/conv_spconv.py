"""The sparse convolutions on spconv (CUDA kernels compiled for sm_61 and up): the backend of the cards below sm_80,
where flex_gemm's Triton kernels are not used (modules/precision.py).

The weight and the bias are this module's own parameters, with the names and the layout flex_gemm's backend gives
them (`weight` [Co, Kd, Kh, Kw, Ci] - spconv's KRSC layout is the same - and `bias` [Co]), so that the checkpoints
load unchanged whatever the backend. The spconv module is kept aside, unregistered (its parameters would otherwise
appear under `conv.` in the state dict), and lent the parameters at every call: the loader replaces the Parameter
objects when it assigns the weights (load_state_dict(assign=True)), so a one-off alias would go stale.

spconv's GEMM kernels are prebuilt for sm_60, 70, 75, 80, 86, 89 and 90 only; on another card (a GTX 10xx is sm_61)
spconv compiles them at run time through NVRTC, which cannot work from the wheel: no headers are shipped and no
include path is given ("could not open source file tensorview/core/all.h", a GTX 1080). Such a card runs spconv's
Native algorithm with the products on torch.mm, one GEMM per kernel offset (`indice_conv_torch`, put in place of
spconv's ops.indice_conv, which its module reaches for subm, strided and inverse convolutions alike).
"""
from typing import Optional

import torch
import torch.nn as nn
from .. import SparseTensor
from ... import precision
from . import config
import spconv.pytorch as spconv
from spconv.pytorch import ops as spconv_ops


def _own(self, conv) -> None:
    """Take `conv`'s weight and bias as this module's parameters and keep `conv` aside."""
    want = (conv.out_channels, *conv.kernel_size, conv.in_channels)
    if tuple(conv.weight.shape) != tuple(want):
        raise RuntimeError(f"spconv's weight layout {tuple(conv.weight.shape)} is not KRSC {tuple(want)}")
    self.weight = nn.Parameter(conv.weight.detach())
    if conv.bias is not None:
        self.bias = nn.Parameter(conv.bias.detach())
    else:
        self.register_parameter("bias", None)
    conv._parameters.pop("weight", None)
    conv._parameters.pop("bias", None)
    object.__setattr__(self, "_conv", conv)
    self.in_channels, self.out_channels = conv.in_channels, conv.out_channels


def _lent(self):
    """The spconv module with this module's parameters in place."""
    conv = self._conv
    conv._parameters["weight"] = self.weight
    conv._parameters["bias"] = self.bias
    return conv


def indice_conv_torch(features, filters, indice_pairs, indice_pair_num, num_activate_out, inverse=False, subm=False,
                      algo=None, timer=None, bias=None, act_alpha=0.0, act_beta=0.0, act_type=None):
    """spconv's gather - product - scatter convolution (ops.indice_conv, its Native algorithm) with the products on
    torch.mm: for every kernel offset, the rows that offset pairs times the offset's [Ci, Co] slice of the KRSC
    weight, summed into the output rows (fp32 rounding order apart, spconv's kernels' sums)."""
    if act_type is not None and act_type != spconv_ops.tv.gemm.Activation.None_:
        raise NotImplementedError("a fused activation")
    co, ci = filters.shape[0], filters.shape[-1]
    w = filters.reshape(co, -1, ci)                         # [Co, kv, Ci]
    kv = w.shape[1]
    centre = kv // 2
    counts = indice_pair_num.tolist()                       # the host sync spconv's own path has
    pair_in, pair_out = indice_pairs[int(inverse)], indice_pairs[int(not inverse)]
    out = torch.mm(features, w[:, centre].t()) if subm else features.new_zeros((num_activate_out, co))
    for i, n in enumerate(counts):
        if subm and i == centre:
            continue
        if subm and i > centre:                             # a subm's far half mirrors the near half's pairs
            n = counts[kv - i - 1]
        if n <= 0:
            continue
        rows = features.index_select(0, pair_in[i, :n].long())
        out.index_add_(0, pair_out[i, :n].long(), torch.mm(rows, w[:, i].t()))
    if bias is not None:
        out += bias
    return out


_torch_products: Optional[bool] = None


def _products_in_torch() -> bool:
    """True on a card whose spconv GEMM kernels are not prebuilt (the module's docstring): the first call puts
    indice_conv_torch in place of spconv's ops.indice_conv, which spconv's functional module looks up at each call."""
    global _torch_products
    if _torch_products is None:
        from spconv.core_cc.cumm.common import CompileInfo
        cc = precision.profile().capability
        _torch_products = cc is not None and not CompileInfo.arch_is_compiled_gemm(tuple(cc))
        if _torch_products:
            spconv_ops.indice_conv = indice_conv_torch
    return _torch_products


def _algo():
    if config.SPCONV_ALGO == 'native' or _products_in_torch():
        return spconv.ConvAlgo.Native
    if config.SPCONV_ALGO == 'implicit_gemm':
        return spconv.ConvAlgo.MaskImplicitGemm
    return None


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    algo = _algo()
    if stride == 1 and (padding is None):
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias, indice_key=indice_key, algo=algo)
    else:
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, indice_key=indice_key, algo=algo)
    _own(self, conv)
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)
    self.padding = padding


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    spatial_changed = any(s != 1 for s in self.stride) or (self.padding is not None)
    new_data = _lent(self)(x.data)
    new_shape = [x.shape[0], self.out_channels]
    new_layout = None if spatial_changed else x.layout

    if spatial_changed and (x.shape[0] != 1):
        # spconv was non-1 stride will break the contiguous of the output tensor, sort by the coords
        fwd = new_data.indices[:, 0].argsort()
        bwd = torch.zeros_like(fwd).scatter_(0, fwd, torch.arange(fwd.shape[0], device=fwd.device))
        sorted_feats = new_data.features[fwd]
        sorted_coords = new_data.indices[fwd]
        unsorted_data = new_data
        new_data = spconv.SparseConvTensor(sorted_feats, sorted_coords, unsorted_data.spatial_shape, unsorted_data.batch_size)  # type: ignore

    out = SparseTensor(
        new_data, shape=torch.Size(new_shape), layout=new_layout,
        scale=tuple([s * stride for s, stride in zip(x._scale, self.stride)]),
        spatial_cache=x._spatial_cache,
    )

    if spatial_changed and (x.shape[0] != 1):
        out.register_spatial_cache(f'conv_{self.stride}_unsorted_data', unsorted_data)
        out.register_spatial_cache(f'conv_{self.stride}_sort_bwd', bwd)

    return out


def sparse_inverse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
    _own(self, spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, bias=bias, indice_key=indice_key))
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    spatial_changed = any(s != 1 for s in self.stride)
    if spatial_changed:
        # recover the original spconv order
        data = x.get_spatial_cache(f'conv_{self.stride}_unsorted_data')
        bwd = x.get_spatial_cache(f'conv_{self.stride}_sort_bwd')
        data = data.replace_feature(x.feats[bwd])
    else:
        data = x.data

    new_data = _lent(self)(data)
    new_shape = [x.shape[0], self.out_channels]
    new_layout = None if spatial_changed else x.layout
    out = SparseTensor(
        new_data, shape=torch.Size(new_shape), layout=new_layout,
        scale=tuple([s // stride for s, stride in zip(x._scale, self.stride)]),
        spatial_cache=x._spatial_cache,
    )
    return out
