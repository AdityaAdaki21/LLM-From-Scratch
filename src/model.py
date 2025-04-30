# src/model.py
import torch
import torch.nn as nn
import math
from . import config # Relative import

class PositionalEncoding(nn.Module):
    # (Keep the PositionalEncoding class definition exactly as before)
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        # Ensure pe is on the correct device during initialization if needed,
        # but register_buffer handles device placement during model.to(device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # Ensure pe slice matches x's seq_len and is on the same device as x
        x = x + self.pe[:x.size(0)].to(x.device)
        return self.dropout(x)


class SimpleTransformerBlock(nn.Module):
     # (Keep the SimpleTransformerBlock class definition exactly as before)
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        # batch_first=True is important for data shape consistency
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # Consider using GELU for potentially better performance like many modern LLMs
        self.activation = nn.ReLU() # Or nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src shape expected: [batch_size, seq_len, d_model] due to batch_first=True
        src_norm = self.norm1(src)
        # MultiHeadAttention expects query, key, value.
        # src_mask prevents attending to future tokens (causal mask).
        # src_key_padding_mask prevents attending to padding tokens.
        attn_output, _ = self.self_attn(src_norm, src_norm, src_norm,
                                         attn_mask=src_mask,
                                         key_padding_mask=src_key_padding_mask,
                                         need_weights=False) # Set to True if you want to inspect attention weights
        src = src + self.dropout1(attn_output) # Residual connection

        # Feed Forward part
        src_norm2 = self.norm2(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src_norm2))))
        src = src + self.dropout2(ff_output) # Residual connection
        return src

class SimpleDecoderLLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Pass max_seq_len from config here
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)
        self.transformer_layers = nn.ModuleList([
            SimpleTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        # LayerNorm and Linear layers inside TransformerBlock usually have decent default init

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        # Ensure the mask is created on the same device as the input tensors
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, src: torch.Tensor, src_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_padding_mask: Tensor, shape [batch_size, seq_len], True indicates padding
        """
        # 1. Embedding + Positional Encoding
        # src shape: [batch_size, seq_len]
        embedded_src = self.embedding(src) * math.sqrt(self.d_model) # shape: [batch_size, seq_len, d_model]
        # PositionalEncoding expects [seq_len, batch, dim], so transpose needed
        pos_encoded_src = self.pos_encoder(embedded_src.transpose(0, 1)).transpose(0, 1) # shape: [batch_size, seq_len, d_model]

        # 2. Generate causal mask (needs to be on the correct device)
        seq_len = src.size(1)
        causal_mask = self._generate_square_subsequent_mask(seq_len, src.device) # Pass device

        # 3. Pass through transformer layers
        output = pos_encoded_src
        for layer in self.transformer_layers:
            output = layer(output, src_mask=causal_mask, src_key_padding_mask=src_padding_mask)

        # 4. Final normalization and output layer
        output = self.norm(output)
        logits = self.fc_out(output) # shape: [batch_size, seq_len, vocab_size]
        return logits

# --- Helper function to instantiate model based on config and tokenizer ---
def create_model(tokenizer):
    vocab_size = tokenizer.get_vocab_size()
    print(f"Creating model with Vocab Size: {vocab_size}")
    model = SimpleDecoderLLM(
        vocab_size=vocab_size,
        d_model=config.D_MODEL,
        nhead=config.N_HEADS,
        num_layers=config.N_LAYERS,
        dim_feedforward=config.D_FF,
        max_seq_len=config.MAX_SEQ_LEN, # Pass max_seq_len here
        dropout=config.DROPOUT
    )
    print(f"Model created with ~{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    return model