"""Native causal conv1d for GDN (session-state, no paging).

conv_state layout: [num_seqs, conv_dim, kernel_size - 1]  (channel-last friendly)
"""
import torch
import torch.nn.functional as F


def causal_conv1d_prefill(x, weight, bias, conv_state, activation="silu"):
    """
    x: [seqlen, conv_dim]  (single sequence tokens, time-major)
    weight: [conv_dim, kernel]
    bias: [conv_dim] or None
    conv_state: [conv_dim, kernel-1]  (in/out, updated in-place)
    returns: [seqlen, conv_dim]
    """
    seqlen, conv_dim = x.shape
    kernel = weight.shape[1]
    # build [conv_dim, kernel-1 + seqlen] : prepend conv_state
    xt = x.transpose(0, 1)                       # [conv_dim, seqlen]
    inp = torch.cat([conv_state, xt], dim=1)     # [conv_dim, kernel-1+seqlen]
    # depthwise conv1d
    out = F.conv1d(
        inp.unsqueeze(0),                        # [1, conv_dim, L]
        weight.unsqueeze(1),                     # [conv_dim, 1, kernel]
        bias=bias,
        groups=conv_dim,
    ).squeeze(0)                                 # [conv_dim, seqlen]
    # update conv_state = last (kernel-1) columns of inp
    new_state = inp[:, -(kernel - 1):].contiguous()
    conv_state.copy_(new_state)
    out = out.transpose(0, 1)                    # [seqlen, conv_dim]
    if activation in ("silu", "swish"):
        out = F.silu(out)
    return out


def causal_conv1d_decode(x, weight, bias, conv_state, activation="silu"):
    """
    x: [num_seqs, conv_dim]  (one token per seq)
    weight: [conv_dim, kernel]
    conv_state: [num_seqs, conv_dim, kernel-1]  (in/out, updated in-place)
    returns: [num_seqs, conv_dim]
    """
    num_seqs, conv_dim = x.shape
    kernel = weight.shape[1]
    # window = [conv_state | x] : [num_seqs, conv_dim, kernel]
    window = torch.cat([conv_state, x.unsqueeze(-1)], dim=-1)   # [S, conv_dim, kernel]
    out = (window * weight.unsqueeze(0)).sum(dim=-1)            # [S, conv_dim]
    if bias is not None:
        out = out + bias
    # shift conv_state left by 1
    conv_state.copy_(window[:, :, 1:].contiguous())
    if activation in ("silu", "swish"):
        out = F.silu(out)
    return out