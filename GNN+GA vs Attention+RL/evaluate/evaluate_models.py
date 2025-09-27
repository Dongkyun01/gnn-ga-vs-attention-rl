import os
import argparse
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from models.gnn_ga import GNNGAModel
from models.attention_rl import AttentionRLModel
from utils.tsp_utils import compute_distance_matrix, compute_tour_length, nearest_neighbor
from utils.ga_utils import run_ga


# -----------------------------
# Evaluate GNN+GA
# -----------------------------
def evaluate_gnn_ga(model, coords, device, ga_pop=20, ga_gen=200):
    coords_t = torch.tensor(coords, dtype=torch.float32).to(device)
    probs = model(coords_t)  # (N,)
    probs_np = probs.detach().cpu().numpy()

    # 확률 순서대로 초기 투어 생성
    order = np.argsort(-probs_np).tolist()
    unique_order = []
    for node in order:
        if node not in unique_order:
            unique_order.append(node)
    if len(unique_order) < len(order):
        for node in range(len(coords)):
            if node not in unique_order:
                unique_order.append(node)

    tour = unique_order + [unique_order[0]]
    dist_matrix = compute_distance_matrix(coords)

    # 초기 population
    init_population = [tour]
    for _ in range(ga_pop - 1):
        nn_tour = nearest_neighbor(dist_matrix, start=np.random.randint(len(coords)))
        init_population.append(nn_tour)

    best_tour, best_length = run_ga(dist_matrix, init_population, generations=ga_gen)
    return best_tour, best_length


# -----------------------------
# Evaluate Attention+RL
# -----------------------------
def evaluate_attention_rl(model, coords, device):
    coords_t = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(device)  # (1, N, 2)
    mask = torch.zeros(1, coords_t.size(1), device=device)
    tour = []

    for step in range(coords_t.size(1)):
        probs = model(coords_t, mask)  # (1, N)
        next_node = torch.argmax(probs, dim=-1).item()
        tour.append(next_node)
        mask[0, next_node] = 1

    order = tour + [tour[0]]
    dist_matrix = compute_distance_matrix(coords)
    length = compute_tour_length(order, dist_matrix)
    return order, length


# -----------------------------
# Main Evaluation Loop
# -----------------------------
def evaluate_models(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    gnn_ga = GNNGAModel(in_dim=2, hidden_dim=128, n_layers=3).to(device)
    gnn_ga.load_state_dict(torch.load(args.gnn_model, map_location=device))
    gnn_ga.eval()

    attn_rl = AttentionRLModel(embed_dim=128, num_heads=8, num_layers=3).to(device)
    attn_rl.load_state_dict(torch.load(args.attn_model, map_location=device))
    attn_rl.eval()

    # 테스트셋 리스트
    test_sets = {
        "in_distribution": "data/test_uniform.npy",
        "scale": "data/test_scale.npy",
        "cluster": "data/test_cluster.npy",
        "circle": "data/test_circle.npy"
    }

    results = {}
    all_sample_records = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for name, path in test_sets.items():
        data = np.load(path, allow_pickle=True)
        lengths_gnn = []
        lengths_attn = []
        success_count = 0
        per_sample_records = []

        for idx, coords in enumerate(tqdm(data[:args.n_samples], desc=f"Evaluating {name}")):
            gnn_tour, L_gnn = evaluate_gnn_ga(gnn_ga, coords, device, ga_pop=args.ga_pop, ga_gen=args.ga_gen)
            attn_tour, L_attn = evaluate_attention_rl(attn_rl, coords, device)

            lengths_gnn.append(L_gnn)
            lengths_attn.append(L_attn)

            gap = L_gnn - L_attn
            if L_attn < L_gnn:
                success_count += 1

            record = {
                "dataset": name,
                "sample_id": idx,
                "coords": coords.tolist(),       # ✅ 좌표 저장
                "gnn_ga_tour": gnn_tour,         # ✅ GA 경로 저장
                "gnn_ga_length": L_gnn,
                "attn_rl_tour": attn_tour,       # ✅ RL 경로 저장
                "attn_rl_length": L_attn,
                "gap": gap
            }
            per_sample_records.append(record)
            all_sample_records.append(record)

        # dataset별 요약
        results[name] = {
            "gnn_ga_mean": np.mean(lengths_gnn),
            "gnn_ga_std": np.std(lengths_gnn),
            "attn_rl_mean": np.mean(lengths_attn),
            "attn_rl_std": np.std(lengths_attn),
            "success_rate": success_count / len(per_sample_records)
        }

        # dataset별 샘플 CSV 저장
        per_sample_df = pd.DataFrame(per_sample_records)
        os.makedirs("results", exist_ok=True)
        sample_csv_path = f"results/eval_samples_{name}_{timestamp}.csv"
        per_sample_df.to_csv(sample_csv_path, index=False)
        print(f"  📑 Saved per-sample results: {sample_csv_path}")

    # 요약 CSV 저장
    summary_csv_path = f"results/eval_summary_{timestamp}.csv"
    summary_df = pd.DataFrame([
        {
            "dataset": name,
            "gnn_ga_mean": res["gnn_ga_mean"],
            "gnn_ga_std": res["gnn_ga_std"],
            "attn_rl_mean": res["attn_rl_mean"],
            "attn_rl_std": res["attn_rl_std"],
            "success_rate": res["success_rate"],
            "n_samples": args.n_samples,
            "ga_pop": args.ga_pop,
            "ga_gen": args.ga_gen,
        }
        for name, res in results.items()
    ])
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n✅ Summary saved to {summary_csv_path}")

    # 전체 샘플 통합 CSV 저장
    all_csv_path = f"results/eval_all_samples_{timestamp}.csv"
    all_samples_df = pd.DataFrame(all_sample_records)
    all_samples_df.to_csv(all_csv_path, index=False)
    print(f"✅ All sample results saved to {all_csv_path}")

    # 결과 출력
    print("\n=== Evaluation Results ===")
    for name, res in results.items():
        print(f"[{name}]")
        print(f"  GNN+GA: mean={res['gnn_ga_mean']:.4f}, std={res['gnn_ga_std']:.4f}")
        print(f"  Attn+RL: mean={res['attn_rl_mean']:.4f}, std={res['attn_rl_std']:.4f}")
        print(f"  Success Rate (Attn+RL better): {res['success_rate']*100:.2f}%")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn_model", type=str, default="checkpoints/gnn_ga_model.pt")
    parser.add_argument("--attn_model", type=str, default="checkpoints/attention_rl_model.pt")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--ga_pop", type=int, default=20)
    parser.add_argument("--ga_gen", type=int, default=200)
    args = parser.parse_args()

    evaluate_models(args)
