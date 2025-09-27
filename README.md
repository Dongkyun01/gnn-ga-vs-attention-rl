#  GNN+GA vs Attention+RL for TSP

## 1. 연구 배경 및 계획
본 프로젝트는 **Travelling Salesman Problem (TSP)** 를 대상으로,  
- **GNN+GA (Graph Neural Network + Genetic Algorithm)**  
- **Attention+RL (Self-Attention + Reinforcement Learning)**  

두 가지 접근법을 비교·분석하는 것을 목표로 한다.  

### 연구 목적
- **RQ1.** 동일 분포(in-distribution)에서 학습한 후, 테스트셋에서 두 모델의 성능 차이는 어떠한가?  
- **RQ2.** 학습과 다른 분포(스케일 변화, 클러스터, 원형 구조)에서 두 모델의 일반화 성능은 어떠한가?  
- **RQ3.** GA(전역 탐색 보강)와 RL(정책 학습) 중 어떤 방식이 더 안정적이고 일관된 해를 제공하는가?  

---

## 2. 가설 (Hypotheses)
1. **Attention+RL은 학습 분포(in-distribution)에서 GNN+GA보다 더 우수할 것이다.**  
2. **GNN+GA는 다른 분포(out-of-distribution)에서 더 안정적일 것이다.**  
3. **원형 구조(circle)와 같은 규칙적 패턴에서는 Attention+RL이 압도적으로 강할 것이다.**  

---

## 3. 모델 구성

### 3.1 GNN+GA
- Encoder: **Graph Attention Network (GAT)**  
- Decoder: **Pointer Network**  
- 학습 방식: **첫 move만 teacher forcing**  
- 추가 처리: 학습 후 **Genetic Algorithm**으로 경로 최적화  

### 3.2 Attention+RL
- Encoder: **Multi-head Self-Attention (Transformer 기반)**  
- Policy 학습: **REINFORCE with baseline (greedy tour)**  
- 탐색 방식: **확률적 샘플링 (Stochastic decoding)**  

---

## 4. 데이터셋 구성
- **훈련 데이터**: Uniform 분포 (랜덤 좌표, 50 nodes)  
- **테스트 데이터 (OOD)**:  
  - **Scale**: 좌표 범위 확장  
  - **Cluster**: 클러스터 구조 생성  
  - **Circle**: 원형 분포  

---

## 5. 하이퍼파라미터
| 항목 | 값 |
|------|----------------|
| Train size | 30,000 |
| Test size | 1,000 |
| Epochs | 30 |
| Batch size (GNN+GA) | 1024 |
| Batch size (Attn+RL) | 512 |
| Optimizer | Adam (lr=1e-4) |
| GA 설정 | population=20, generations=200 |

---

## 6. 실험 결과

### 6.1 In-distribution
- GNN+GA: mean=15.54, std=1.08  
- Attn+RL: mean=9.57, std=1.21  
 **Attn+RL 압도적 성능 (성공률 100%)**

---

### 6.2 Scale
- GNN+GA: mean=154.61, std=11.78  
- Attn+RL: mean=175.86, std=31.97  
 **GNN+GA가 더 안정적**

---

### 6.3 Cluster
- GNN+GA: mean=79.91, std=21.59  
- Attn+RL: mean=87.18, std=35.60  
**GNN+GA 우세 (클러스터 구조에서 GA의 효과)**

---

### 6.4 Circle
- GNN+GA: mean=36.70, std=2.53  
- Attn+RL: mean=10.32, std=0.00  
**Attn+RL 압도적 (정책이 deterministic하게 수렴, 모든 시도에서 동일 해 → std=0)**  

---

## 7. 결론 (Discussion & Conclusion)
- **Attn+RL**
  - 학습 분포 및 규칙성이 강한 구조(원형)에서 압도적 성능
  - Circle 데이터에서 variance=0 → **정책이 완벽히 수렴**

- **GNN+GA**
  - 학습 분포를 벗어난 Scale·Cluster 환경에서 더 안정적
  - GA 후처리(local search)가 분포 변화에 강건성을 제공

### 핵심 인사이트
- In-distribution → **RL 승**  
- Out-of-distribution → **GNN+GA 승**  
- 구조적 규칙(원형) → **RL 완승**  

> 따라서, **실무 적용**에서는  
> - 데이터 분포가 고정적/규칙적일 경우 → **RL 기반 방법**  
> - 데이터 분포가 다양하고 변화 가능성이 높을 경우 → **GNN+GA 기반 방법**  
이 더 적합하다.

---

## 8. 실행 방법 (Quick Start)


# 데이터 생성
python main_experiment.py --make_data --train_size 30000 --test_size 1000

# GNN+GA 학습
python main_experiment.py --train_gnn --train_size 30000 --gnn_epochs 30 --gnn_batch_size 1024 --seed 42

# Attention+RL 학습
python main_experiment.py --train_attn --train_size 30000 --attn_epochs 30 --attn_batch_size 512 --seed 42

# 평가 및 시각화
python main_experiment.py --evaluate --visualize --eval_samples 200 --ga_pop 20 --ga_gen 200

## 9. 폴더 구조

```bash
GNN+GA vs Attention+RL
├── data/                    
│   ├── __init__.py
│   └── generate_tsp_data.py
├── dataset/                 
├── evaluate/
│   ├── __init__.py
│   └── evaluate_models.py   
│   ├── __init__.py
│   ├── attention_rl.py      # Attention+RL 모델
│   └── gnn_ga.py            # GNN+GA 모델
├── trainers/
│   ├── __init__.py
│   ├── train_attention_rl.py
│   └── train_gnn_ga.py
├── utils/
│   └── tsp_utils.py
├── main_experiment.py        # 전체 파이프라인 실행 스크립트
├── visualize_results.py      # 평가 결과 시각화
└── visualize_routes.py       # 경로 시각화

```

## 10. 참고 문헌

Bello, I., Pham, H., Le, Q. V., Norouzi, M., & Bengio, S. (2017). Neural Combinatorial Optimization with Reinforcement Learning. arXiv preprint arXiv:1611.09940.

Kool, W., van Hoof, H., & Welling, M. (2019). Attention, Learn to Solve Routing Problems!. International Conference on Learning Representations (ICLR).

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2017). Graph Attention Networks. arXiv preprint arXiv:1710.10903.

Kovács, L., & Jlidi, A. (2024). Neural Networks for Vehicle Routing Problem. Advanced Logistic Systems – Theory and Practice, 18(2), 17-29. 
