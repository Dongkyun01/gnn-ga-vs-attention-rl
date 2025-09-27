import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import pandas as pd   # ✅ 로그 저장용

from models.attention_rl import AttentionRLModel
from utils.tsp_utils import compute_distance_matrix, compute_tour_length


# -----------------------------
# Tour 샘플링 (stochastic decoding)
# -----------------------------
def sample_tour(model, coords, device):
    B, N, _ = coords.shape
    mask = torch.zeros(B, N, device=device)
    tours = [[] for _ in range(B)]

    for step in range(N):
        probs = model(coords, mask)  # (B, N)
        dist = torch.distributions.Categorical(probs)
        next_node = dist.sample()  # stochastic 선택

        for b in range(B):
            tours[b].append(next_node[b].item())

        mask[torch.arange(B), next_node] = 1

    lengths = []
    for b in range(B):
        order = tours[b] + [tours[b][0]]
        dist_matrix = compute_distance_matrix(coords[b].cpu().numpy())
        L = compute_tour_length(order, dist_matrix)
        lengths.append(L)

    return tours, torch.tensor(lengths, dtype=torch.float32, device=device)


# -----------------------------
# Greedy Tour (baseline)
# -----------------------------
def greedy_tour(model, coords, device):
    B, N, _ = coords.shape
    mask = torch.zeros(B, N, device=device)
    tours = [[] for _ in range(B)]

    for step in range(N):
        probs = model(coords, mask)  # (B, N)
        next_node = torch.argmax(probs, dim=-1)

        for b in range(B):
            tours[b].append(next_node[b].item())

        mask[torch.arange(B), next_node] = 1

    lengths = []
    for b in range(B):
        order = tours[b] + [tours[b][0]]
        dist_matrix = compute_distance_matrix(coords[b].cpu().numpy())
        L = compute_tour_length(order, dist_matrix)
        lengths.append(L)

    return tours, torch.tensor(lengths, dtype=torch.float32, device=device)


# -----------------------------
# Training Loop (REINFORCE with baseline)
# -----------------------------
def train(model, optimizer, train_data, device, epochs=10, batch_size=64,
          out_dir="checkpoints", start_epoch=0, epoch_losses=[]):
    model.train()

    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        n_batches = 0  # ✅ 배치 수 카운트

        for i in tqdm(range(0, len(train_data), batch_size)):
            batch = train_data[i:i+batch_size]
            coords = torch.tensor(batch, dtype=torch.float32).to(device)  # (B, N, 2)

            # (1) Stochastic 샘플링
            tours, lengths = sample_tour(model, coords, device)

            # (2) Greedy baseline
            _, baseline_lengths = greedy_tour(model, coords, device)

            # (3) Advantage = reward - baseline
            reward = -lengths
            baseline = -baseline_lengths
            advantage = reward - baseline

            # (4) Policy gradient loss
            log_probs = []
            mask = torch.zeros(coords.size(0), coords.size(1), device=device)

            for step in range(coords.size(1)):
                probs = model(coords, mask)
                dist = torch.distributions.Categorical(probs)
                actions = [tour[step] for tour in tours]
                actions = torch.tensor(actions, device=device)
                log_prob = dist.log_prob(actions)
                log_probs.append(log_prob.unsqueeze(1))
                mask[torch.arange(coords.size(0)), actions] = 1

            log_probs = torch.cat(log_probs, dim=1).sum(dim=1)  # (B,)
            loss = -(advantage.detach() * log_probs).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # ✅ 평균 Loss 계산 (배치 기준)
        avg_loss = total_loss / n_batches
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # ✅ 에포크마다 모델 저장
        os.makedirs(out_dir, exist_ok=True)
        epoch_model_path = os.path.join(out_dir, f"attention_rl_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), epoch_model_path)

        # ✅ 마지막 체크포인트 저장 (resume 용)
        last_ckpt = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "losses": epoch_losses
        }
        torch.save(last_ckpt, os.path.join(out_dir, "attention_rl_last.pt"))
        print(f"[Checkpoint Saved] {epoch_model_path}, attention_rl_last.pt")

    return epoch_losses


# -----------------------------
# Main
# -----------------------------
def main(args):
    # ✅ seed 고정
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로드
    train_data = np.load(args.data_path, allow_pickle=True)
    if args.n_samples:
        train_data = train_data[:args.n_samples]

    # 모델 초기화
    model = AttentionRLModel(embed_dim=128, num_heads=8, num_layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    start_epoch = 0
    epoch_losses = []

    # ✅ resume 옵션
    if args.resume:
        ckpt_path = os.path.join(args.out_dir, "attention_rl_last.pt")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = checkpoint["epoch"]
            epoch_losses = checkpoint["losses"]
            print(f"[Resume] Loaded checkpoint from {ckpt_path}, resuming at epoch {start_epoch+1}")
        else:
            print("⚠️ No checkpoint found, starting from scratch.")

    # 학습
    epoch_losses = train(model, optimizer, train_data, device,
                         epochs=args.epochs, batch_size=args.batch_size,
                         out_dir=args.out_dir, start_epoch=start_epoch,
                         epoch_losses=epoch_losses)

    # ✅ 최종 모델 저장
    os.makedirs(args.out_dir, exist_ok=True)
    final_model_path = os.path.join(args.out_dir, "attention_rl_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"[Saved] Final Attention+RL model at {final_model_path}")

    # ✅ 로그 CSV 저장
    os.makedirs("results", exist_ok=True)
    log_path = f"results/train_attn_logs_seed{args.seed}.csv"
    df = pd.DataFrame({
        "epoch": list(range(1, len(epoch_losses) + 1)),
        "loss": epoch_losses,
        "seed": args.seed
    })
    df.to_csv(log_path, index=False)
    print(f"[Saved] Training log at {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train_uniform.npy", help="훈련 데이터 경로")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="모델 저장 경로")
    parser.add_argument("--epochs", type=int, default=10, help="학습 epoch 수")
    parser.add_argument("--batch_size", type=int, default=64, help="배치 크기")
    parser.add_argument("--n_samples", type=int, default=1000, help="훈련 데이터 샘플 수 (디버깅용)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 (재현성 확인용)")
    parser.add_argument("--resume", action="store_true", help="마지막 체크포인트에서 이어서 학습")
    args = parser.parse_args()

    main(args)
