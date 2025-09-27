import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.gnn_ga import GNNGAModel
from models.attention_rl import AttentionRLModel
from utils.tsp_utils import compute_distance_matrix, compute_tour_length, nearest_neighbor
from utils.ga_utils import run_ga


# -----------------------------
# GNN+GA로 투어 생성
# -----------------------------
def get_gnn_ga_tour(model, coords, device, ga_pop=20, ga_gen=200):
    coords_t = torch.tensor(coords, dtype=torch.float32).to(device)
    probs = model(coords_t)  # (N,)
    probs_np = probs.detach().cpu().numpy()

    # 초기 투어 생성
    order = np.argsort(-probs_np).tolist()
    unique_order = []
    for node in order:
        if node not in unique_order:
            unique_order.append(node)
    for node in range(len(coords)):
        if node not in unique_order:
            unique_order.append(node)
    tour = unique_order + [unique_order[0]]

    dist_matrix = compute_distance_matrix(coords)
    init_population = [tour]
    for _ in range(ga_pop - 1):
        nn_tour = nearest_neighbor(dist_matrix, start=np.random.randint(len(coords)))
        init_population.append(nn_tour)

    best_tour, best_length = run_ga(dist_matrix, init_population, generations=ga_gen)
    return best_tour, best_length


# -----------------------------
# Attention+RL로 투어 생성
# -----------------------------
def get_attention_rl_tour(model, coords, device):
    coords_t = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.zeros(1, coords_t.size(1), device=device)
    tour = []

    for step in range(coords_t.size(1)):
        probs = model(coords_t, mask)
        next_node = torch.argmax(probs, dim=-1).item()
        tour.append(next_node)
        mask[0, next_node] = 1

    order = tour + [tour[0]]
    dist_matrix = compute_distance_matrix(coords)
    length = compute_tour_length(order, dist_matrix)
    return order, length


# -----------------------------
# 경로 시각화
# -----------------------------
def plot_tour(coords, tour, title, out_path):
    plt.figure(figsize=(6, 6))
    coords = np.array(coords)

    # 도시 점 찍기
    plt.scatter(coords[:, 0], coords[:, 1], c="blue", zorder=2)

    # 경로 연결
    path_coords = coords[tour]
    plt.plot(path_coords[:, 0], path_coords[:, 1], "-o", c="orange", zorder=1)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn_model", type=str, default="checkpoints/gnn_ga_model.pt")
    parser.add_argument("--attn_model", type=str, default="checkpoints/attention_rl_model.pt")
    parser.add_argument("--dataset", type=str, default="data/test_uniform.npy")
    parser.add_argument("--n_samples", type=int, default=3, help="몇 개 샘플을 그릴지")
    parser.add_argument("--out_dir", type=str, default="figures/routes")
    parser.add_argument("--ga_pop", type=int, default=20)
    parser.add_argument("--ga_gen", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    gnn_ga = GNNGAModel(in_dim=2, hidden_dim=128, n_layers=3).to(device)
    gnn_ga.load_state_dict(torch.load(args.gnn_model, map_location=device))
    gnn_ga.eval()

    attn_rl = AttentionRLModel(embed_dim=128, num_heads=8, num_layers=3).to(device)
    attn_rl.load_state_dict(torch.load(args.attn_model, map_location=device))
    attn_rl.eval()

    # 데이터 로드
    data = np.load(args.dataset, allow_pickle=True)

    for i, coords in enumerate(data[:args.n_samples]):
        # GNN+GA
        gnn_tour, gnn_len = get_gnn_ga_tour(gnn_ga, coords, device, args.ga_pop, args.ga_gen)
        plot_tour(coords, gnn_tour, f"GNN+GA (len={gnn_len:.2f})", os.path.join(args.out_dir, f"sample{i}_gnn.png"))

        # Attention+RL
        attn_tour, attn_len = get_attention_rl_tour(attn_rl, coords, device)
        plot_tour(coords, attn_tour, f"Attention+RL (len={attn_len:.2f})", os.path.join(args.out_dir, f"sample{i}_attn.png"))

        print(f"✅ Sample {i}: GNN+GA={gnn_len:.2f}, Attn+RL={attn_len:.2f}")
