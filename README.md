# COVID-19 Fake News Detection Using Graph Neural Networks

## 1. Problem Definition  
COVID-19 관련 가짜뉴스가 확산되며 신체적, 정신적 피해를 야기하는 문제를 해결하기 위해, **Graph Neural Network (GNN)**를 활용한 **Node Classification**으로 가짜뉴스 탐지를 목표로 합니다.

---

## 2. Data  
**Dataset**: `train.csv`, `test.csv`  
**Features**:  
- `id`: claim id  
- `label`: 0 (normal) / 1 (fake)  
- `published_date`: 보도 날짜  
- `keybert_keywords`: keybert로 추출한 키워드  
- `ner_keywords`: ner로 추출한 키워드  
- `youtube0 ~ youtube9`: 유튜브 검색 상위 10개 텍스트  

**Pre-processing**:  
- **Graph 생성**: TF-IDF 기반 코사인 유사도로 엣지 구성  
- **텍스트 임베딩**: BERT 활용  
- **추가 피처**: 클레임 길이, 감정 점수 (TextBlob 사용)  

---

## 3. Model  
- **알고리즘**: Graph Attention Network (GAT)  
- **구성**:  
  - GATConv 레이어, Dropout(0.4), Batch Normalization  
  - 임베딩 결합: 클레임 + 키워드 임베딩  
  - 학습률 스케줄링: StepLR  

---

## 4. Experiments  
- **Train/Test Split**: 80/20  
- **Optimizer**: AdamW + StepLR  
- **Metrics**: 정확도, 교차 엔트로피 손실  

---

## 5. Results & Discussion  
- **결과**: 에포크마다 훈련/검증 손실 및 정확도 모니터링, 조기 종료 적용  
- **활용**: 텍스트 기반 진위 판별 (뉴스, 리뷰, 소셜 미디어 등)  
- **한계**: 고차원 임베딩으로 인한 높은 학습 비용, 그래프 구조 설정 민감성  

---

## References  
- [GAT 논문](https://arxiv.org/abs/1710.10903)  
- [TextBlob](https://textblob.readthedocs.io/en/dev/)  
- [BERT Word Embedding](https://riverkangg.github.io/nlp/nlp-bertWordEmbedding/)  
