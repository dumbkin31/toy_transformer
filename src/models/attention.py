import torch
import torch.nn as nn
import math

# using boolean mask
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, torch.finfo(attn_scores.dtype).min)

        attn_probs = torch.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_probs, V)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.scaled_dot_product_attention = ScaledDotProductAttention()
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def rotate_half(self, x):
        """Splits the last dimension in half and rotates it."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_emb(self, x, cos, sin):
        """
        Applies RoPE to a tensor x of shape [batch, heads, seq_len, head_dim].
        cos, sin shape: [seq_len, head_dim]
        """
        # 1. Align cos/sin dimensions to match x: [1, 1, seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
        
        # 2. Slice cos/sin to match the exact sequence length of x (handles causal/cross attention)
        seq_len = x.shape[2]
        cos = cos[:, :, :seq_len, :]
        sin = sin[:, :, :seq_len, :]
        
        # 3. Apply the rotation formula: (X * cos) + (rotate_half(X) * sin)
        return (x * cos) + (self.rotate_half(x) * sin)
    
    def forward(self, x_q, x_k, x_v, mask=None, rope=False):
        Q = self.split_heads(self.W_q(x_q))
        K = self.split_heads(self.W_k(x_k))
        V = self.split_heads(self.W_v(x_v))

        if rope:
            pass
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()

        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.d_k = d_model // num_heads
        self.num_groups = num_heads // num_kv_heads

        self.W_q = nn.Linear(d_model, d_model)

        # K and V only need num_kv_heads
        self.W_k = nn.Linear(
            d_model,
            num_kv_heads * self.d_k
        )

        self.W_v = nn.Linear(
            d_model,
            num_kv_heads * self.d_k
        )

        self.W_o = nn.Linear(d_model, d_model)

        self.scaled_dot_product_attention = ScaledDotProductAttention()

    def forward(self):
        pass