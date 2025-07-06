# 비만 예측 머신러닝 프로젝트 (Obesity Prediction Machine Learning Project)

## 📋 프로젝트 개요

이 프로젝트는 개인의 생활습관, 식습관, 신체적 특성 등을 기반으로 비만도를 예측하는 머신러닝 모델들을 구현한 프로젝트입니다. 다양한 분류 알고리즘을 사용하여 비만도를 7단계로 분류하고, 각 모델의 성능을 비교 분석합니다.

## 🎯 목표

- 개인의 건강 관련 데이터를 활용한 비만도 예측
- 다양한 머신러닝 알고리즘의 성능 비교
- 지도학습과 비지도학습의 결합 방법 탐구
- 실용적인 비만 예측 시스템 구축

## 📊 데이터셋

**ObesityDataSet_raw_and_data_sinthetic_BMI.csv**
- 비만도 분류: 7단계 (Insufficient_Weight, Normal_Weight, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III, Overweight_Level_I, Overweight_Level_II)
- 특성: 성별, 나이, 가족력, 식습관, 운동습관, 생활습관 등 14개 변수

## 🔧 구현된 모델들

### 1. 기본 분류 모델들

#### **Bagging.ipynb**
- **알고리즘**: Bagging (Bootstrap Aggregating)
- **기본 분류기**: Decision Tree
- **정확도**: 97.4%
- **특징**: 앙상블 기법으로 과적합 방지

#### **Boosting.ipynb**
- **알고리즘**: AdaBoost
- **기본 분류기**: Decision Tree
- **정확도**: 96.9%
- **특징**: 순차적 학습으로 약한 분류기를 강화

#### **KNN_classif.ipynb**
- **알고리즘**: K-Nearest Neighbors
- **K값**: 3
- **정확도**: 95.4%
- **특징**: 비모수적 방법, 간단하고 직관적

#### **Navie_bayes_classif.ipynb**
- **알고리즘**: Gaussian Naive Bayes
- **정확도**: 80.6%
- **특징**: 확률 기반 분류, 빠른 학습 속도

#### **RandomForest.ipynb**
- **알고리즘**: Random Forest
- **트리 개수**: 100
- **정확도**: 97.9%
- **특징**: 가장 높은 성능, 특성 중요도 분석 가능

### 2. 고급 분석 모델들

#### **지도학습_only.ipynb**
- **알고리즘**: Random Forest (200개 트리)
- **정확도**: 79.2%
- **특징**:
  - 전체 데이터와 BMI ≥ 25 데이터 분리 분석
  - 특성 중요도 시각화
  - 혼동 행렬 분석
  - 사용자 입력 기반 실시간 예측
  - 점수 기반 순위 시스템

#### **지도after비지도.ipynb**
- **알고리즘**: K-means + Random Forest
- **정확도**: 85.1%
- **특징**:
  - 비지도학습(K-means)으로 클러스터링 후 지도학습
  - PCA를 이용한 3D 시각화
  - 실루엣 점수 분석
  - 클러스터 정보를 추가 특성으로 활용

## 📈 성능 비교

| 모델 | 정확도 | 특징 |
|------|--------|------|
| Random Forest | 97.9% | 최고 성능, 특성 중요도 분석 |
| Bagging | 97.4% | 과적합 방지 효과 |
| AdaBoost | 96.9% | 순차적 학습 |
| KNN | 95.4% | 간단하고 직관적 |
| 지도after비지도 | 85.1% | 클러스터링 정보 활용 |
| 지도학습_only | 79.2% | 실용적 기능 포함 |
| Naive Bayes | 80.6% | 빠른 학습 속도 |

## 🛠️ 기술 스택

- **Python**: 주요 프로그래밍 언어
- **Scikit-learn**: 머신러닝 라이브러리
- **Pandas**: 데이터 처리
- **NumPy**: 수치 계산
- **Matplotlib/Seaborn**: 데이터 시각화
- **Jupyter Notebook**: 개발 환경

## 🚀 주요 기능

### 1. 모델 성능 평가
- 정확도 (Accuracy)
- 분류 보고서 (Precision, Recall, F1-score)
- 혼동 행렬 (Confusion Matrix)
- 특성 중요도 분석

### 2. 데이터 시각화
- 3D PCA 시각화
- 특성 중요도 막대 그래프
- 혼동 행렬 히트맵

### 3. 실시간 예측
- 사용자 입력 기반 비만도 예측
- 점수 기반 순위 시스템
- BMI 계산 및 분석

## 📁 파일 구조

```
Find_Obesity_MachineLearning_Project/
├── Bagging.ipynb              # Bagging 앙상블 모델
├── Boosting.ipynb             # AdaBoost 모델
├── KNN_classif.ipynb          # K-Nearest Neighbors 모델
├── Navie_bayes_classif.ipynb  # Naive Bayes 모델
├── RandomForest.ipynb         # Random Forest 모델
├── 지도after비지도.ipynb      # 비지도+지도학습 결합 모델
└── 지도학습_only.ipynb        # 고급 지도학습 분석
```

## 🎯 결론 및 인사이트

1. **Random Forest가 가장 우수한 성능**을 보이며, 특성 중요도 분석도 가능
2. **앙상블 기법들(Bagging, Boosting)**이 일반적으로 높은 성능을 보임
3. **비지도학습과 지도학습의 결합**이 새로운 접근 방법으로 효과적
4. **KNN과 Naive Bayes**는 간단하지만 상대적으로 낮은 성능
5. **실용적인 기능들**이 추가된 모델이 실제 활용도가 높음

## 🔮 향후 개선 방향

- 딥러닝 모델 적용 (Neural Networks)
- 하이퍼파라미터 튜닝 최적화
- 더 많은 특성 엔지니어링
- 실시간 웹 애플리케이션 구축
- 모바일 앱 개발

## 📝 사용법

1. Jupyter Notebook 환경에서 각 `.ipynb` 파일 실행
2. 데이터셋 파일이 같은 디렉토리에 있는지 확인
3. 필요한 라이브러리 설치:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn numpy
   ```
