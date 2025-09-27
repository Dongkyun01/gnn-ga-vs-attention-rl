import os
import glob
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set(style="whitegrid", font_scale=1.2)

# -----------------------------
# 1. 학습 곡선
# -----------------------------
def plot_training_curves(results_dir="results"):
    gnn_logs = sorted(glob.glob(os.path.join(results_dir, "train_gnn_logs_seed*.csv")))
    attn_logs = sorted(glob.glob(os.path.join(results_dir, "train_attn_logs_seed*.csv")))

    plt.figure(figsize=(10, 6))
    for log in gnn_logs:
        df = pd.read_csv(log)
        plt.plot(df["epoch"], df["loss"], label=f"GNN+GA ({os.path.basename(log)})")
    for log in attn_logs:
        df = pd.read_csv(log)
        plt.plot(df["epoch"], df["loss"], label=f"Attn+RL ({os.path.basename(log)})")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plot_training_curves.png"))
    plt.close()
    print("✅ Saved: plot_training_curves.png")


# -----------------------------
# 2. 평가 요약 (bar + error bar)
# -----------------------------
def plot_eval_summary(results_dir="results"):
    summary_files = sorted(glob.glob(os.path.join(results_dir, "eval_summary_*.csv")))
    if not summary_files:
        print("⚠️ No eval_summary CSV found.")
        return
    df = pd.read_csv(summary_files[-1])  # 최신 파일

    df_melted = pd.melt(df,
                        id_vars=["dataset"],
                        value_vars=["gnn_ga_mean", "attn_rl_mean"],
                        var_name="Model", value_name="TourLength")

    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_melted, x="dataset", y="TourLength", hue="Model", errorbar="sd")
    plt.title("Evaluation Summary (Mean Tour Length)")
    plt.ylabel("Tour Length")
    plt.savefig(os.path.join(results_dir, "plot_eval_summary.png"))
    plt.close()
    print("✅ Saved: plot_eval_summary.png")


# -----------------------------
# 3. 성능 분포 (박스플롯)
# -----------------------------
def plot_eval_distributions(results_dir="results"):
    all_files = sorted(glob.glob(os.path.join(results_dir, "eval_all_samples_*.csv")))
    if not all_files:
        print("⚠️ No eval_all_samples CSV found.")
        return
    df = pd.read_csv(all_files[-1])  # 최신 파일

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="dataset", y="gnn_ga_length", color="skyblue")
    sns.boxplot(data=df, x="dataset", y="attn_rl_length", color="salmon")
    plt.title("Tour Length Distributions per Dataset")
    plt.ylabel("Tour Length")
    plt.savefig(os.path.join(results_dir, "plot_eval_distributions.png"))
    plt.close()
    print("✅ Saved: plot_eval_distributions.png")


# -----------------------------
# 4. 성공률 (Win Rate)
# -----------------------------
def plot_win_rates(results_dir="results"):
    summary_files = sorted(glob.glob(os.path.join(results_dir, "eval_summary_*.csv")))
    if not summary_files:
        print("⚠️ No eval_summary CSV found.")
        return
    df = pd.read_csv(summary_files[-1])

    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="dataset", y="success_rate", color="green")
    plt.title("Success Rate (Attn+RL better than GNN+GA)")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(results_dir, "plot_win_rates.png"))
    plt.close()
    print("✅ Saved: plot_win_rates.png")


# -----------------------------
# 5. 경로 시각화 (좌표 + 투어 비교)
# -----------------------------
def plot_sample_tours(results_dir="results", dataset="in_distribution", sample_id=0):
    sample_files = sorted(glob.glob(os.path.join(results_dir, f"eval_samples_{dataset}_*.csv")))
    if not sample_files:
        print(f"⚠️ No eval_samples CSV found for dataset={dataset}")
        return
    df = pd.read_csv(sample_files[-1])  # 최신 파일

    if sample_id >= len(df):
        print(f"⚠️ sample_id {sample_id} is out of range.")
        return

    row = df.iloc[sample_id]
    coords = np.array(ast.literal_eval(row["coords"]))
    gnn_tour = ast.literal_eval(row["gnn_ga_tour"])
    attn_tour = ast.literal_eval(row["attn_rl_tour"])

    plt.figure(figsize=(8, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c="black", s=30, label="Cities")

    # GNN+GA 경로
    gnn_coords = coords[gnn_tour]
    plt.plot(gnn_coords[:, 0], gnn_coords[:, 1], "-o", color="blue", alpha=0.7, label="GNN+GA")

    # Attn+RL 경로
    attn_coords = coords[attn_tour]
    plt.plot(attn_coords[:, 0], attn_coords[:, 1], "-o", color="red", alpha=0.7, label="Attn+RL")

    plt.title(f"Sample {sample_id} ({dataset})\nGNN+GA vs Attn+RL")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(results_dir, f"plot_sample_tour_{dataset}_{sample_id}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved: {save_path}")


# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":
    results_dir = "results"
    plot_training_curves(results_dir)
    plot_eval_summary(results_dir)
    plot_eval_distributions(results_dir)
    plot_win_rates(results_dir)
    # 샘플 경로 예시
    plot_sample_tours(results_dir, dataset="in_distribution", sample_id=0)
