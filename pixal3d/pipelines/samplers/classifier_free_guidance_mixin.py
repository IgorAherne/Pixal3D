from typing import *

# The pair of passes of a guided step runs one x_t at one t, with the context and without. The first block's
# self-attention half (norm1, the attention, its gated residual) reads no context, so it gives the same bits in
# both: the second pass starts from the stream the first one kept (the flows' blocks, modules/sparse/transformer and
# modules/transformer modulated.py, _self_half_shared). One [tokens, channels] table is kept from the first block of
# the first pass to the first block of the second.
SHARE_FIRST_BLOCK = True


def _first_block(model):
    blocks = getattr(model, "blocks", None)
    if not SHARE_FIRST_BLOCK or blocks is None or len(blocks) == 0:
        return None
    first = blocks[0]
    return first if hasattr(first, "self_attn") else None


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, guidance_strength, guidance_rescale=0.0, **kwargs):
        if guidance_strength == 1:
            return super()._inference_model(model, x_t, t, cond, **kwargs)
        if neg_cond is None:
            raise ValueError(f"neg_cond is needed when guidance_strength is not 1 (it is {guidance_strength})")
        if guidance_strength == 0:
            return super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        else:
            first = _first_block(model)
            pair = {"keep": True}
            try:
                if first is not None:
                    first.cfg_pair = pair
                pred_pos = super()._inference_model(model, x_t, t, cond, **kwargs)
                pair["keep"] = False
                pred_neg = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            finally:
                pair.clear()
                if first is not None:
                    first.cfg_pair = None
            pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg

            # CFG rescale
            if guidance_rescale > 0:
                x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                x_0_cfg = self._pred_to_xstart(x_t, t, pred)
                std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                pred = self._xstart_to_pred(x_t, t, x_0)

            return pred
