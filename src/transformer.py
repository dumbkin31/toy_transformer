import torch
import torch.nn as nn

from models.attention import MultiHeadAttention, GroupedQueryAttention
from models.norm import LayerNorm, RMSNorm
from models.ffn import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        if config.attention == "gqa":
            self.attention = GroupedQueryAttention(self.d_model, config.num_heads)
        else:
            self.attention = MultiHeadAttention(self.d_model, config.num_heads)

        if config.layernorm == "rms":
            self.norm1 = RMSNorm(self.d_model)
            self.norm2 = RMSNorm(self.d_model)
        else:
            self.norm1 = LayerNorm(self.d_model)
            self.norm2 = LayerNorm(self.d_model)

        self.ffn = PositionWiseFeedForward(self.d_model, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask):
        norm_x = self.norm1(x)
        attn_output = self.attention(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(attn_output)

        norm_x = self.norm2(x)
        ff_output = self.ffn(norm_x)
        x = x + self.dropout(ff_output)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        if config.attention == "gqa":
            self.self_attn = GroupedQueryAttention(self.d_model, config.num_heads)
            self.cross_attn = GroupedQueryAttention(self.d_model, config.num_heads)
        else:
            self.self_attn = MultiHeadAttention(self.d_model, config.num_heads)
            self.cross_attn = MultiHeadAttention(self.d_model, config.num_heads)

        if config.layernorm == "rms":
            self.norm1 = RMSNorm(self.d_model)
            self.norm2 = RMSNorm(self.d_model)
            self.norm3 = RMSNorm(self.d_model)
        else:
            self.norm1 = LayerNorm(self.d_model)
            self.norm2 = LayerNorm(self.d_model)
            self.norm3 = LayerNorm(self.d_model)

        self.ffn = PositionWiseFeedForward(self.d_model, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Masked self-attention
        norm_x = self.norm1(x)
        attn_output = self.self_attn(norm_x, norm_x, norm_x, tgt_mask)
        x = x + self.dropout(attn_output)

        # Cross-attention
        norm_x = self.norm2(x)
        attn_output = self.cross_attn(norm_x, enc_output, enc_output, src_mask)
        x = x + self.dropout(attn_output)

        # Feed-forward
        norm_x = self.norm3(x)
        ff_output = self.ffn(norm_x)
        x = x + self.dropout(ff_output)

        return x

class Transformer(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass