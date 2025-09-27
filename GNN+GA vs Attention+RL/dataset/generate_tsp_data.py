import os
import argparse
import numpy as np


# -----------------------------
# 1. Uniform 분포 데이터
# -----------------------------
def generate_uniform(n_nodes=50, scale=1.0):
    """[0,scale]x[0,scale] 범위에서 uniform random 노드 좌표 생성"""
    return np.random.rand(n_nodes, 2) * scale


# -----------------------------
# 2. Gaussian Cluster 데이터
# -----------------------------
def generate_cluster(n_nodes=50, n_clusters=3, scale=10.0):
    """n_clusters 개의 Gaussian cluster 분포 노드 좌표 생성"""
    cluster_centers = np.random.rand(n_clusters, 2) * scale
    points = []
    for i in range(n_nodes):
        center = cluster_centers[np.random.randint(n_clusters)]
        point = np.random.normal(loc=center, scale=scale/20, size=(2,))
        points.append(point)
    return np.array(points)


# -----------------------------
# 3. Circle 데이터
# -----------------------------
def generate_circle(n_nodes=50, radius=1.0):
    """원 위에 균일하게 분포된 노드 좌표 생성"""
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    points = np.stack([radius*np.cos(angles), radius*np.sin(angles)], axis=1)
    return points


# -----------------------------
# 데이터 저장 함수
# -----------------------------
def save_dataset(data_list, out_path):
    """데이터셋을 numpy 배열로 저장"""
    data_arr = np.array(data_list)
    np.save(out_path, data_arr)
    print(f"[Saved] {out_path}, shape={data_arr.shape}")


# -----------------------------
# 메인 실행
# -----------------------------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 훈련 데이터 (Uniform [0,1])
    train_data = [generate_uniform(n_nodes=args.n_nodes, scale=1.0) 
                  for _ in range(args.train_size)]
    save_dataset(train_data, os.path.join(args.out_dir, "train_uniform.npy"))

    # 2. 테스트 데이터
    # (a) In-distribution [0,1]
    test_uniform = [generate_uniform(n_nodes=args.n_nodes, scale=1.0) 
                    for _ in range(args.test_size)]
    save_dataset(test_uniform, os.path.join(args.out_dir, "test_uniform.npy"))

    # (b) Out-of-distribution [0,10]
    test_scale = [generate_uniform(n_nodes=args.n_nodes, scale=10.0) 
                  for _ in range(args.test_size)]
    save_dataset(test_scale, os.path.join(args.out_dir, "test_scale.npy"))

    # (c) Out-of-distribution Cluster
    test_cluster = [generate_cluster(n_nodes=args.n_nodes, scale=10.0) 
                    for _ in range(args.test_size)]
    save_dataset(test_cluster, os.path.join(args.out_dir, "test_cluster.npy"))

    # (d) Out-of-distribution Circle
    test_circle = [generate_circle(n_nodes=args.n_nodes, radius=1.0) 
                   for _ in range(args.test_size)]
    save_dataset(test_circle, os.path.join(args.out_dir, "test_circle.npy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data", help="저장 경로")
    parser.add_argument("--n_nodes", type=int, default=50, help="TSP 노드 개수")
    parser.add_argument("--train_size", type=int, default=100000, help="훈련 데이터 개수")
    parser.add_argument("--test_size", type=int, default=1000, help="테스트 데이터 개수")
    args = parser.parse_args()

    main(args)
