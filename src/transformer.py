import torch
import torch.nn as nn

from models.attention import MultiHeadAttention, GroupedQueryAttention
from models.norm import LayerNorm, RMSNorm
from models.ffn import PositionWiseFeedForward
from models.positional import PositionalEncoding

class TransformerConfig:
    def __init__(
            self, 
            d_model=128, 
            d_ff=512, 
            max_seq_len=352, 
            num_heads=4, 
            num_layers=2,
            rope=False, 
            attention="mha", 
            normalization="layernorm",
            dropout=0.1,
            src_vocab_size=2000,
            tgt_vocab_size=5000
        ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.rope = rope
        self.attention = attention
        self.layernorm = normalization
        self.dropout = dropout

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.rope = config.rope
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
        attn_output = self.attention(norm_x, norm_x, norm_x, mask, self.rope)
        x = x + self.dropout(attn_output)

        norm_x = self.norm2(x)
        ff_output = self.ffn(norm_x)
        x = x + self.dropout(ff_output)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.rope = config.rope
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
        attn_output = self.self_attn(norm_x, norm_x, norm_x, tgt_mask, self.rope)
        x = x + self.dropout(attn_output)

        # Cross-attention
        norm_x = self.norm2(x)
        attn_output = self.cross_attn(norm_x, enc_output, enc_output, src_mask, self.rope)
        x = x + self.dropout(attn_output)

        # Feed-forward
        norm_x = self.norm3(x)
        ff_output = self.ffn(norm_x)
        x = x + self.dropout(ff_output)

        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rope = config.rope
        self.encoder_embedding = nn.Embedding(config.src_vocab_size, config.d_model)
        self.decoder_embedding = nn.Embedding(config.tgt_vocab_size, config.d_model)
        if not config.rope:
            self.positional_encoding = PositionalEncoding(config.d_model, config.max_seq_len)

        self.encoder_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])

        if config.layernorm == "rms":
            self.enc_norm = RMSNorm(config.d_model)
            self.dec_norm = RMSNorm(config.d_model)
        else:
            self.enc_norm = LayerNorm(config.d_model)
            self.dec_norm = LayerNorm(config.d_model)

        self.fc = nn.Linear(config.d_model, config.tgt_vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def generate_mask(self, src, tgt):
        # Ignore padding tokens in the source (mask keys)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_len)

        # Ignore padding tokens in the target — mask KEYS, not queries
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)

        # Prevent decoder from looking at future tokens
        seq_length = tgt.size(1)
        nopeak_mask = torch.tril(
            torch.ones(seq_length, seq_length, dtype=torch.bool, device=tgt.device)
        )

        tgt_mask = tgt_pad_mask & nopeak_mask  # broadcasts to (batch, 1, tgt_len, tgt_len)

        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.encoder_embedding(src)
        tgt_embedded = self.decoder_embedding(tgt)

        if not self.rope:
            src_embedded = self.positional_encoding(src_embedded)
            tgt_embedded = self.positional_encoding(tgt_embedded)

        src_embedded = self.dropout(src_embedded)
        tgt_embedded = self.dropout(tgt_embedded)

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        enc_output = self.enc_norm(enc_output)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        dec_output = self.dec_norm(dec_output)

        output = self.fc(dec_output)
        return output