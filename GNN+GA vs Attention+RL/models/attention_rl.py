import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Multi-Head Attention
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, D)
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(x + self.dropout(attn_output))
        return x


# -----------------------------
# Feed Forward Network
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=512, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = self.linear2(self.dropout(x))
        return self.norm(residual + self.dropout(x))


# -----------------------------
# Transformer Encoder
# -----------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, num_layers=3, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                MultiHeadAttention(embed_dim, num_heads, dropout),
                FeedForward(embed_dim, hidden_dim=512, dropout=dropout)
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# -----------------------------
# Pointer Decoder
# -----------------------------
class PointerDecoder(nn.Module):
    def __init__(self, embed_dim=128):
        super(PointerDecoder, self).__init__()
        self.context = nn.Parameter(torch.randn(embed_dim))  # 초기 context
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, node_embeddings, mask=None):
        """
        node_embeddings: (B, N, D)
        mask: (B, N) 방문 여부 (방문하면 -inf)
        return: (B, N) 방문 확률
        """
        B, N, D = node_embeddings.shape
        context = self.context.expand(B, 1, D)  # (B, 1, D)
        scores = torch.bmm(node_embeddings, self.proj(context).transpose(1, 2)).squeeze(-1)  # (B, N)

        if mask is not None:
            scores = scores.masked_fill(mask == 1, -1e9)

        probs = F.softmax(scores, dim=-1)
        return probs


# -----------------------------
# 전체 모델 (Encoder + Decoder)
# -----------------------------
class AttentionRLModel(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, num_layers=3, dropout=0.1):
        super(AttentionRLModel, self).__init__()
        self.embedding = nn.Linear(2, embed_dim)  # 좌표 → 임베딩
        self.encoder = TransformerEncoder(embed_dim, num_heads, num_layers, dropout)
        self.decoder = PointerDecoder(embed_dim)

    def forward(self, coords, mask=None):
        """
        coords: (B, N, 2) 노드 좌표
        mask: (B, N) 방문 여부
        return: (B, N) 방문 확률
        """
        x = self.embedding(coords)  # (B, N, D)
        enc = self.encoder(x)       # (B, N, D)
        probs = self.decoder(enc, mask)  # (B, N)
        return probs
