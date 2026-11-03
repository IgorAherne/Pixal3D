from typing import *

CONV = 'flex_gemm' 
DEBUG = False
ATTN = 'flash_attn'
SAGE = False        # PIXAL3D_SAGE: one sequence's full self-attention through SageAttention (modules/sage.py)

def __from_env():
    import os
    
    global CONV
    global DEBUG
    global ATTN
    global SAGE
    
    env_sparse_conv_backend = os.environ.get('SPARSE_CONV_BACKEND')
    env_sparse_debug = os.environ.get('SPARSE_DEBUG')
    env_sparse_attn_backend = os.environ.get('SPARSE_ATTN_BACKEND')
    if env_sparse_attn_backend is None:
        env_sparse_attn_backend = os.environ.get('ATTN_BACKEND')

    if env_sparse_conv_backend is not None and env_sparse_conv_backend in ['none', 'spconv', 'torchsparse', 'flex_gemm']:
        CONV = env_sparse_conv_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == '1'
    if env_sparse_attn_backend is not None and env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'flash_attn_4', 'sdpa']:
        ATTN = env_sparse_attn_backend

    # a backend without kernels for this card is replaced by the card's (modules/precision.py)
    from .. import precision, sage
    ATTN = precision.attention_backend(ATTN)
    CONV = precision.conv_backend(CONV, explicit=env_sparse_conv_backend is not None)
    SAGE = sage.from_env()

    print(f"[SPARSE] Conv backend: {CONV}; Attention backend: {ATTN}"
          + ("; one sequence's full self-attention: SageAttention (PIXAL3D_SAGE)" if SAGE else ""))
        

__from_env()
    

def set_conv_backend(backend: Literal['none', 'spconv', 'torchsparse', 'flex_gemm']):
    global CONV
    CONV = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

def set_attn_backend(backend: Literal['xformers', 'flash_attn', 'flash_attn_3', 'flash_attn_4', 'sdpa']):
    global ATTN
    ATTN = backend

def set_sage(on: bool):
    global SAGE
    SAGE = bool(on)
