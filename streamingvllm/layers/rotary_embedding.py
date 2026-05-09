from functools import lru_cache
import torch
from torch import nn

def apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, rotary_dim, max_position_embeddings, base,
                 mrope_section_sizes=None):
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)
        self.mrope_section_sizes = mrope_section_sizes

        # Precompute MRoPE masks for O(1) cache lookup (CUDA Graph & Compile safe)
        half_dim = head_size // 2
        mask_h = torch.zeros(half_dim, dtype=torch.bool)
        mask_w = torch.zeros(half_dim, dtype=torch.bool)
        
        if mrope_section_sizes is not None:
            sections = list(mrope_section_sizes)
        else:
            s = half_dim // 3
            sections = [s, s, half_dim - 2 * s]
            
        length_h = sections[1] * 3
        idx_h = list(range(1, min(length_h, half_dim), 3))
        mask_h[idx_h] = True
        
        length_w = sections[2] * 3
        idx_w = list(range(2, min(length_w, half_dim), 3))
        mask_w[idx_w] = True
        
        # Expand to full head_dim (for both cos and sin)
        mask_h_full = torch.cat([mask_h, mask_h], dim=-1).view(1, 1, -1)
        mask_w_full = torch.cat([mask_w, mask_w], dim=-1).view(1, 1, -1)
        
        self.register_buffer("mask_h_full", mask_h_full, persistent=False)
        self.register_buffer("mask_w_full", mask_w_full, persistent=False)

    @torch.compile
    def _forward_standard(self, positions, query, key):
        """Standard 1D RoPE - fully compiled"""
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)

    @torch.compile
    def _forward_mrope(self, positions, query, key):
        """
        O(1) MRoPE using precomputed masks and cache lookups.
        Fully compatible with torch.compile and CUDA Graphs.
        """
        c_s_t = self.cos_sin_cache[positions[0]]
        c_s_h = self.cos_sin_cache[positions[1]]
        c_s_w = self.cos_sin_cache[positions[2]]

        # Blend T, H, W frequencies using precomputed masks
        c_s = torch.where(self.mask_h_full, c_s_h, c_s_t)
        c_s = torch.where(self.mask_w_full, c_s_w, c_s)

        cos, sin = c_s.chunk(2, dim=-1)
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)

    def forward(self, positions, query, key):
        if positions.dim() == 2:
            return self._forward_mrope(positions, query, key)
        return self._forward_standard(positions, query, key)

@lru_cache(1)
def get_rope(head_size, rotary_dim, max_position, base, mrope_section_sizes=None):
    return RotaryEmbedding(head_size, rotary_dim, max_position, base,
                           mrope_section_sizes=mrope_section_sizes)
