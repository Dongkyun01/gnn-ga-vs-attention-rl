import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import pandas as pd   # 로그 저장용

from models.gnn_ga import GNNGAModel
from utils.tsp_utils import compute_distance_matrix, nearest_neighbor, two_opt


# -----------------------------
# Teacher (NN + 2-opt)
# -----------------------------
def generate_teacher_tour(coords):
    dist_matrix = compute_distance_matrix(coords)
    nn_tour = nearest_neighbor(dist_matrix, start=0)
    opt_tour = two_opt(nn_tour, dist_matrix)
    return opt_tour  # 예: [0, 3, 1, 2, 4, 0]


# -----------------------------
# Dataset Loader
# -----------------------------
def load_dataset(path, n_samples=None):
    data = np.load(path, allow_pickle=True)
    if n_samples:
        data = data[:n_samples]
    return data


# -----------------------------
# Training Loop (첫 move만 학습)
# -----------------------------
def train(model, optimizer, train_data, device, epochs=10, batch_size=64,
          out_dir="checkpoints", start_epoch=0, epoch_losses=[]):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        n_batches = 0

        for i in tqdm(range(0, len(train_data), batch_size)):
            batch = train_data[i:i+batch_size]

            optimizer.zero_grad()
            batch_loss = 0.0

            for coords in batch:
                coords = torch.tensor(coords, dtype=torch.float32).to(device)
                teacher_tour = generate_teacher_tour(coords.cpu().numpy())
                teacher_tour = torch.tensor(teacher_tour[:-1], dtype=torch.long, device=device)

                # 🔹 첫 move만 teacher forcing
                probs = model(coords, tour=teacher_tour)  # (N,)
                target = teacher_tour[0]  # 첫 move 정답
                loss = criterion(probs.unsqueeze(0), target.unsqueeze(0))

                batch_loss += loss

            # 역전파 & 최적화
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            n_batches += 1

        # 🔹 배치 평균 loss
        avg_loss = total_loss / n_batches
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # 에포크마다 모델 저장
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, f"gnn_ga_epoch{epoch+1}.pt"))

        # 마지막 체크포인트 저장 (resume용)
        last_ckpt = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "losses": epoch_losses
        }
        torch.save(last_ckpt, os.path.join(out_dir, "gnn_ga_last.pt"))

    return epoch_losses


# -----------------------------
# Main
# -----------------------------
def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로드
    train_data = load_dataset(args.data_path, n_samples=args.n_samples)

    # 모델 초기화
    model = GNNGAModel(in_dim=2, hidden_dim=128, n_layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    start_epoch = 0
    epoch_losses = []

    # ✅ resume 옵션
    if args.resume:
        ckpt_path = os.path.join(args.out_dir, "gnn_ga_last.pt")
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

    # 최종 모델 저장
    os.makedirs(args.out_dir, exist_ok=True)
    final_model_path = os.path.join(args.out_dir, "gnn_ga_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"[Saved] Final GNN+GA model at {final_model_path}")

    # 로그 CSV 저장
    os.makedirs("results", exist_ok=True)
    log_path = f"results/train_gnn_logs_seed{args.seed}.csv"
    df = pd.DataFrame({
        "epoch": list(range(1, len(epoch_losses) + 1)),
        "loss": epoch_losses,
        "seed": args.seed
    })
    df.to_csv(log_path, index=False)
    print(f"[Saved] Training log at {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train_uniform.npy")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    main(args)
