import numpy as np


# -----------------------------
# 거리 계산 함수 = 거리행렬 변환
# -----------------------------
def compute_distance_matrix(coords):
    """
    coords: (N, 2) 좌표
    """
    N = len(coords)
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    return dist_matrix


def compute_tour_length(tour, dist_matrix):
    """
    tour: [노드 순서 리스트], 마지막은 시작 노드로 복귀해야 함
    """
    length = 0.0
    for i in range(len(tour) - 1):
        u = int(tour[i])       # 인덱스 강제 int 변환
        v = int(tour[i + 1])
        dist = dist_matrix[u, v]
        length += float(dist)  # numpy.float32 → python float
    return float(length)



# -----------------------------
# Nearest Neighbor 휴리스틱
# -----------------------------
def nearest_neighbor(dist_matrix, start=0):
    """
    Nearest Neighbor 휴리스틱으로 초기 경로 생성
    start: 시작 노드
    """
    n = dist_matrix.shape[0]
    visited = [start]
    current = start
    while len(visited) < n:
        # 방문하지 않은 노드 중 가장 가까운 곳 선택
        next_node = min(
            [j for j in range(n) if j not in visited],
            key=lambda j: dist_matrix[current, j]
        )
        visited.append(next_node)
        current = next_node
    visited.append(start)  # 시작점으로 복귀
    return visited


# -----------------------------
# 2-opt Local Search
# -----------------------------
def two_opt(tour, dist_matrix):
    """
    2-opt 알고리즘으로 경로 개선
    tour: 초기 경로 리스트
    """
    best = tour
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i+1, len(tour) - 1):
                if j - i == 1: 
                    continue  # 인접 노드는 교환 안 함
                new_tour = best[:i] + best[i:j][::-1] + best[j:]
                if compute_tour_length(new_tour, dist_matrix) < compute_tour_length(best, dist_matrix):
                    best = new_tour
                    improved = True
        tour = best
    return best
