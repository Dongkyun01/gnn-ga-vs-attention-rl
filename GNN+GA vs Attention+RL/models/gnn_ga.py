import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Graph Attention Layer
# -----------------------------
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h):
        # h: (N, in_dim)
        Wh = self.W(h)  # (N, out_dim)
        N = Wh.size(0)

        Wh_repeat_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeat_alternating = Wh.repeat(N, 1)
        all_combinations = torch.cat([Wh_repeat_in_chunks, Wh_repeat_alternating], dim=1)  # (N*N, 2*out_dim)

        e = self.leakyrelu(self.a(all_combinations)).view(N, N)
        attention = F.softmax(e, dim=1)  # (N, N)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)  # (N, out_dim)
        return h_prime


# -----------------------------
# GNN Encoder
# -----------------------------
class GATEncoder(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, n_layers=3, dropout=0.1):
        super(GATEncoder, self).__init__()
        layers = []
        for i in range(n_layers):
            in_d = in_dim if i == 0 else hidden_dim
            layers.append(GraphAttentionLayer(in_d, hidden_dim, dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = x
        for gat in self.layers:
            h = gat(h)
            h = F.relu(h)
        return h  # (N, hidden_dim)


# -----------------------------
# Pointer Decoder (step-by-step)
# -----------------------------
class PointerDecoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super(PointerDecoder, self).__init__()
        self.context = nn.Parameter(torch.randn(hidden_dim))  # 학습 가능한 초기 context
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_embeddings, mask=None):
        """
        node_embeddings: (N, hidden_dim)
        mask: (N,) 방문 여부 (1=방문됨, 0=미방문)
        """
        context = self.context.unsqueeze(0)  # (1, hidden_dim)
        scores = torch.matmul(node_embeddings, self.proj(context).T).squeeze()  # (N,)

        if mask is not None:
            scores = scores.masked_fill(mask == 1, -1e9)  # 이미 방문한 노드는 제외

        probs = F.softmax(scores, dim=0)  # (N,)
        return probs


# -----------------------------
# 전체 모델 (Encoder + Autoregressive Decoder)
# -----------------------------
class GNNGAModel(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, n_layers=3):
        super(GNNGAModel, self).__init__()
        self.encoder = GATEncoder(in_dim=in_dim, hidden_dim=hidden_dim, n_layers=n_layers)
        self.decoder = PointerDecoder(hidden_dim=hidden_dim)

    def forward(self, coords, tour=None):
        """
        coords: (N, 2)
        tour: Teacher forcing을 위한 정답 투어 (tensor), 없으면 greedy
        return: probs (N,) - 첫 move 확률 분포
        """
        N = coords.size(0)
        node_embeddings = self.encoder(coords)  # (N, hidden_dim)

        mask = torch.zeros(N, device=coords.device)

        # 첫 move 확률 분포 계산
        probs = self.decoder(node_embeddings, mask)  # (N,)

        # teacher forcing: 첫 move 강제 선택
        if tour is not None:
            next_node = tour[0]
        else:
            next_node = torch.argmax(probs).item()

        mask[next_node] = 1

        return probs  # ✅ Tensor 하나만 반환
