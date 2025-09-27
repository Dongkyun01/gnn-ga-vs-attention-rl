import random
import numpy as np
from utils.tsp_utils import compute_tour_length

# -----------------------------
# Fitness 평가
# -----------------------------
def evaluate_population(population, dist_matrix):
    """
    Population 내 각 해의 길이 계산
    population: [tour1, tour2, ...] (tour = 노드 순서 리스트)
    """
    fitness = []
    for tour in population:
        # 안전하게 int 변환 + None 제거
        tour = [int(x) for x in tour if x is not None]
        length = compute_tour_length(tour, dist_matrix)
        fitness.append((tour, length))
    # 길이가 짧을수록 fitness ↑
    return sorted(fitness, key=lambda x: x[1])

# -----------------------------
# Selection (Tournament)
# -----------------------------
def tournament_selection(fitness, k=5):
    """
    Tournament selection
    fitness: [(tour, length), ...] 정렬된 리스트
    """
    candidates = random.sample(fitness, k)
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]  # 가장 좋은 해 반환

# -----------------------------
# Crossover (Order Crossover, OX)
# -----------------------------
def order_crossover(parent1, parent2):
    """
    OX (Order Crossover) 연산
    parent1, parent2: 부모 투어 리스트
    """
    size = len(parent1) - 1  # 마지막은 시작점이므로 제외
    a, b = sorted(random.sample(range(1, size), 2))

    child = [None] * size
    child[a:b] = parent1[a:b]  # 부분 구간 복사

    # parent2의 순서에 따라 남은 도시 채움
    ptr = b
    for node in parent2[1:-1]:  # 끝 노드 제외
        if node not in child:
            if ptr == size:
                ptr = 0
            child[ptr] = node
            ptr += 1

    # ✅ None 값 남아있으면 랜덤으로 보정
    for i in range(size):
        if child[i] is None:
            child[i] = random.randint(0, size - 1)

    return [parent1[0]] + child + [parent1[0]]

# -----------------------------
# Mutation (Swap)
# -----------------------------
def swap_mutation(tour, mutation_rate=0.1):
    """
    Swap mutation: 확률적으로 두 노드 위치 교환
    """
    new_tour = tour[:]
    if random.random() < mutation_rate and len(new_tour) > 3:
        i, j = random.sample(range(1, len(new_tour) - 1), 2)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

# -----------------------------
# Validation (보정)
# -----------------------------
def validate_tour(tour, n_nodes):
    """
    투어가 0~n_nodes-1 범위 안에 있고,
    모든 노드가 정확히 한 번씩 등장하도록 보정
    """
    start = tour[0]
    core = tour[1:-1]

    # ✅ None 제거 + 범위 보정
    core = [int(x) for x in core if x is not None and 0 <= int(x) < n_nodes]

    # 중복 제거
    unique_core = []
    for node in core:
        if node not in unique_core:
            unique_core.append(node)

    # 빠진 노드 보충
    for node in range(n_nodes):
        if node not in unique_core:
            unique_core.append(node)

    return [start] + unique_core + [start]

# -----------------------------
# GA Iteration
# -----------------------------
def run_ga(dist_matrix, init_population, generations=500, mutation_rate=0.1, k=5):
    """
    GA 메인 루프
    dist_matrix: 거리행렬
    init_population: 초기 해 집합 (GNN output or 랜덤)
    """
    n_nodes = dist_matrix.shape[0]
    population = init_population

    for _ in range(generations):
        # Fitness 평가
        fitness = evaluate_population(population, dist_matrix)

        new_population = []
        while len(new_population) < len(population):
            parent1 = tournament_selection(fitness, k)
            parent2 = tournament_selection(fitness, k)
            child = order_crossover(parent1, parent2)
            child = swap_mutation(child, mutation_rate)

            # ✅ 항상 보정 적용
            child = validate_tour(child, n_nodes)
            new_population.append(child)

        population = new_population

    # 최종 해 반환
    fitness = evaluate_population(population, dist_matrix)
    best_tour, best_length = fitness[0]
    return best_tour, best_length



