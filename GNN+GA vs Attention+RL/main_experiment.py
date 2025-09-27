import os
import argparse
import subprocess


def run_command(cmd):
    """터미널 명령어 실행"""
    print(f"\n[Running] {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def main(args):
    # 1. 데이터셋 생성
    if args.make_data:
        run_command(
            f"python -m dataset.generate_tsp_data "
            f"--out_dir data --n_nodes 50 "
            f"--train_size {args.train_size} --test_size {args.test_size}"
        )

    # 2. GNN+GA 학습
    if args.train_gnn:
        cmd = (
            f"python -m trainers.train_gnn_ga "
            f"--data_path data/train_uniform.npy "
            f"--out_dir checkpoints "
            f"--epochs {args.gnn_epochs} "
            f"--batch_size {args.gnn_batch_size} "   # ✅ GNN 전용 batch size
            f"--n_samples {args.train_size} "
            f"--seed {args.seed} "
        )
        if args.resume_gnn:
            cmd += "--resume "
        run_command(cmd)

    # 3. Attention+RL 학습
    if args.train_attn:
        cmd = (
            f"python -m trainers.train_attention_rl "
            f"--data_path data/train_uniform.npy "
            f"--out_dir checkpoints "
            f"--epochs {args.attn_epochs} "
            f"--batch_size {args.attn_batch_size} "  # ✅ Attention 전용 batch size
            f"--n_samples {args.train_size} "
            f"--seed {args.seed} "
        )
        if args.resume_attn:
            cmd += "--resume "
        run_command(cmd)

    # 4. 평가
    if args.evaluate:
        run_command(
            f"python -m evaluate.evaluate_models "
            f"--gnn_model checkpoints/gnn_ga_model.pt "
            f"--attn_model checkpoints/attention_rl_model.pt "
            f"--n_samples {args.eval_samples} "
            f"--ga_pop {args.ga_pop} "
            f"--ga_gen {args.ga_gen}"
        )

    # 5. 시각화
    if args.visualize:
        run_command("python -m visualize_results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--make_data", action="store_true", help="데이터 생성 실행 여부")
    parser.add_argument("--train_gnn", action="store_true", help="GNN+GA 학습 실행 여부")
    parser.add_argument("--train_attn", action="store_true", help="Attention+RL 학습 실행 여부")
    parser.add_argument("--evaluate", action="store_true", help="평가 실행 여부")
    parser.add_argument("--visualize", action="store_true", help="평가 결과 시각화 실행 여부")

    # 공통 설정
    parser.add_argument("--train_size", type=int, default=100000, help="훈련 데이터 개수")
    parser.add_argument("--test_size", type=int, default=1000, help="테스트 데이터 개수")

    # 학습 설정
    parser.add_argument("--gnn_epochs", type=int, default=50, help="GNN 학습 epoch 수")
    parser.add_argument("--attn_epochs", type=int, default=50, help="Attention 학습 epoch 수")
    parser.add_argument("--gnn_batch_size", type=int, default=64, help="GNN 학습 배치 크기")        # ✅ 추가
    parser.add_argument("--attn_batch_size", type=int, default=64, help="Attention 학습 배치 크기")  # ✅ 추가
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")

    # Resume 옵션
    parser.add_argument("--resume_gnn", action="store_true", help="이전 GNN+GA 체크포인트에서 이어서 학습")
    parser.add_argument("--resume_attn", action="store_true", help="이전 Attention+RL 체크포인트에서 이어서 학습")

    # 평가 설정
    parser.add_argument("--eval_samples", type=int, default=200, help="평가에 사용할 샘플 개수")
    parser.add_argument("--ga_pop", type=int, default=20, help="GA population size")
    parser.add_argument("--ga_gen", type=int, default=200, help="GA generation 수")

    args = parser.parse_args()
    main(args)
