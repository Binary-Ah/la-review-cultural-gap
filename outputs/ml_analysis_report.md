# ML & Data Science 분석 보고서

## 데이터 개요
- 장소 통계 행: 17,090
- 업종 통계 행: 242
- 리뷰 전체 건수: 3,602,754
- 리뷰 언어 분포:
  - en: 3,246,685
  - other: 338,271
  - ko: 17,798

## place_stats 주요 지표
- tier 분포: {'fallback': 16702, 'display': 273, 'test': 115}
- 통계적으로 유의미한 장소 수: 53

## category_stats 주요 지표
- reportable 업종 수: 42
- BH-FDR 유의 업종 수: 21

## 가장 큰 언어 평점 격차 장소
### ko가 en보다 훨씬 낮은 장소 (Top 10)
- Painter's Tape (New American restaurant): ko_n=1, en_n=200, ko_mean=1.000, en_mean=4.745, gap=-3.745, tier=fallback
- Tar & Roses (New American restaurant): ko_n=1, en_n=331, ko_mean=1.000, en_mean=4.689, gap=-3.689, tier=fallback
- Kotohira Restaurant (Japanese restaurant): ko_n=1, en_n=125, ko_mean=1.000, en_mean=4.656, gap=-3.656, tier=fallback
- Gardenia Market, Deli & Pizzeria (Restaurant): ko_n=1, en_n=60, ko_mean=1.000, en_mean=4.650, gap=-3.650, tier=fallback
- North End Caffe (Diner): ko_n=1, en_n=150, ko_mean=1.000, en_mean=4.633, gap=-3.633, tier=fallback
- Golden Hunan Chinese Restaurant (Asian restaurant): ko_n=1, en_n=177, ko_mean=1.000, en_mean=4.582, gap=-3.582, tier=fallback
- Pete's Burgers (Fast food restaurant): ko_n=1, en_n=134, ko_mean=1.000, en_mean=4.530, gap=-3.530, tier=fallback
- Azay (Japanese restaurant): ko_n=1, en_n=42, ko_mean=1.000, en_mean=4.476, gap=-3.476, tier=fallback
- Thai Curry Pizza & Thai Cuisine (Pizza restaurant): ko_n=1, en_n=177, ko_mean=1.000, en_mean=4.475, gap=-3.475, tier=fallback
- Flyin Fin Sushi and Seafood (Sushi restaurant): ko_n=1, en_n=142, ko_mean=1.000, en_mean=4.472, gap=-3.472, tier=fallback
### ko가 en보다 훨씬 높은 장소 (Top 10)
- KFC (Chicken restaurant): ko_n=2, en_n=71, ko_mean=5.000, en_mean=2.859, gap=2.141, tier=fallback
- Andre's Pizza (Pizza restaurant): ko_n=1, en_n=51, ko_mean=5.000, en_mean=2.922, gap=2.078, tier=fallback
- Target Glendale Galleria Starbucks (Cafe): ko_n=1, en_n=28, ko_mean=5.000, en_mean=3.000, gap=2.000, tier=fallback
- Starbucks (Coffee shop): ko_n=1, en_n=63, ko_mean=5.000, en_mean=3.016, gap=1.984, tier=fallback
- Sbarro (Pizza restaurant): ko_n=1, en_n=31, ko_mean=5.000, en_mean=3.032, gap=1.968, tier=fallback
- Burger King (Fast food restaurant): ko_n=2, en_n=105, ko_mean=4.500, en_mean=2.590, gap=1.910, tier=fallback
- Starbucks (Coffee shop): ko_n=1, en_n=212, ko_mean=5.000, en_mean=3.104, gap=1.896, tier=fallback
- Popeyes Louisiana Kitchen (Chicken restaurant): ko_n=1, en_n=292, ko_mean=5.000, en_mean=3.144, gap=1.856, tier=fallback
- McDonald's (Fast food restaurant): ko_n=1, en_n=260, ko_mean=5.000, en_mean=3.158, gap=1.842, tier=fallback
- Live Basil Pizza (Pizza restaurant): ko_n=1, en_n=114, ko_mean=5.000, en_mean=3.175, gap=1.825, tier=fallback

## 업종별 상대 격차
### rel_gap가 가장 작은 업종 (ko가 상대적으로 낮은 업종)
- Chocolate cafe: place_n=1, ko_n=1, en_n=51, ko_mean=1.000, en_mean=4.373, abs_gap=-3.373, rel_gap=-3.163
- Ethiopian restaurant: place_n=13, ko_n=1, en_n=1951, ko_mean=2.000, en_mean=4.473, abs_gap=-2.473, rel_gap=-2.263
- Chinese noodle restaurant: place_n=4, ko_n=2, en_n=362, ko_mean=2.000, en_mean=4.232, abs_gap=-2.232, rel_gap=-2.022
- Modern izakaya restaurants: place_n=3, ko_n=3, en_n=309, ko_mean=2.333, en_mean=4.427, abs_gap=-2.094, rel_gap=-1.884
- Cuban restaurant: place_n=21, ko_n=3, en_n=7935, ko_mean=2.667, en_mean=4.401, abs_gap=-1.734, rel_gap=-1.524
- Japanese steakhouse: place_n=1, ko_n=1, en_n=248, ko_mean=3.000, en_mean=4.601, abs_gap=-1.601, rel_gap=-1.391
- Burmese restaurant: place_n=4, ko_n=1, en_n=345, ko_mean=3.000, en_mean=4.458, abs_gap=-1.458, rel_gap=-1.248
- Authentic Japanese restaurant: place_n=12, ko_n=5, en_n=1447, ko_mean=3.000, en_mean=4.355, abs_gap=-1.355, rel_gap=-1.145
- Modern Indian restaurant: place_n=3, ko_n=1, en_n=597, ko_mean=3.000, en_mean=4.310, abs_gap=-1.310, rel_gap=-1.100
- Delivery Restaurant: place_n=10, ko_n=36, en_n=623, ko_mean=2.972, en_mean=4.170, abs_gap=-1.198, rel_gap=-0.988
### rel_gap가 가장 큰 업종 (ko가 상대적으로 높은 업종)
- Cantonese restaurant: place_n=10, ko_n=1, en_n=1000, ko_mean=5.000, en_mean=3.992, abs_gap=1.008, rel_gap=1.218
- Gay bar: place_n=15, ko_n=1, en_n=1382, ko_mean=5.000, en_mean=4.108, abs_gap=0.892, rel_gap=1.102
- Soul food restaurant: place_n=31, ko_n=1, en_n=10812, ko_mean=5.000, en_mean=4.174, abs_gap=0.826, rel_gap=1.035
- Cake shop: place_n=9, ko_n=1, en_n=722, ko_mean=5.000, en_mean=4.208, abs_gap=0.792, rel_gap=1.002
- Snack bar: place_n=6, ko_n=1, en_n=704, ko_mean=5.000, en_mean=4.216, abs_gap=0.784, rel_gap=0.994
- Pizza Takeout: place_n=18, ko_n=3, en_n=1823, ko_mean=5.000, en_mean=4.250, abs_gap=0.750, rel_gap=0.960
- Singaporean restaurant: place_n=3, ko_n=1, en_n=283, ko_mean=5.000, en_mean=4.325, abs_gap=0.675, rel_gap=0.885
- Middle Eastern restaurant: place_n=29, ko_n=3, en_n=5249, ko_mean=5.000, en_mean=4.336, abs_gap=0.664, rel_gap=0.873
- British restaurant: place_n=2, ko_n=2, en_n=1078, ko_mean=5.000, en_mean=4.369, abs_gap=0.631, rel_gap=0.840
- Yakitori restaurant: place_n=2, ko_n=4, en_n=241, ko_mean=5.000, en_mean=4.415, abs_gap=0.585, rel_gap=0.795

## 보고 가능 업종 예시
- Ramen restaurant, Korean restaurant, Asian restaurant, Bakery, Chicken restaurant 등이 reportable로 분류되었습니다.

## ko 리뷰 비중이 높은 업종
- Syokudo and Teishoku restaurant: place_n=1, ko_n=31, en_n=73, ko_mean=4.065, en_mean=3.644, abs_gap=0.421, rel_gap=0.630, ko_rate=0.298
- Korean restaurant: place_n=185, ko_n=4873, en_n=19620, ko_mean=3.884, en_mean=4.338, abs_gap=-0.454, rel_gap=-0.244, ko_rate=0.199
- Shabu-shabu restaurant: place_n=6, ko_n=93, en_n=649, ko_mean=4.032, en_mean=4.274, abs_gap=-0.242, rel_gap=-0.032, ko_rate=0.125
- Korean barbecue restaurant: place_n=80, ko_n=1263, en_n=18825, ko_mean=3.903, en_mean=4.210, abs_gap=-0.306, rel_gap=-0.097, ko_rate=0.063
- Delivery Restaurant: place_n=10, ko_n=36, en_n=623, ko_mean=2.972, en_mean=4.170, abs_gap=-1.198, rel_gap=-0.988, ko_rate=0.055
- Hot pot restaurant: place_n=21, ko_n=57, en_n=2134, ko_mean=4.105, en_mean=4.141, abs_gap=-0.035, rel_gap=0.174, ko_rate=0.026
- Karaoke bar: place_n=18, ko_n=37, en_n=1571, ko_mean=3.757, en_mean=4.024, abs_gap=-0.267, rel_gap=-0.057, ko_rate=0.023
- Pho restaurant: place_n=10, ko_n=35, en_n=1788, ko_mean=4.429, en_mean=4.305, abs_gap=0.124, rel_gap=0.333, ko_rate=0.019
- Hot dog restaurant: place_n=29, ko_n=228, en_n=12191, ko_mean=3.925, en_mean=4.464, abs_gap=-0.539, rel_gap=-0.329, ko_rate=0.018
- Noodle shop: place_n=10, ko_n=44, en_n=2774, ko_mean=3.795, en_mean=4.290, abs_gap=-0.495, rel_gap=-0.285, ko_rate=0.016


---

## 머신러닝 분석 배경 및 의미
- 이 분석은 원본 파이프라인에서 정의한 `lang_group`을 그대로 사용하지만,
  리뷰 텍스트 자체에서 언어 패턴을 학습하는 모델을 만들어 규칙 기반 분류의 보완 가능성을 확인합니다.
- 리뷰가 짧거나 혼합 언어인 경우, 텍스트 기반 모델은 `ko/en/other` 경계 사례를 자동으로 탐지하는 데 도움이 됩니다.

## 모델 학습 및 평가 결과
## 텍스트 기반 언어 분류 모델

### 모델 요약
- 모델: TF-IDF + Logistic Regression
- 클래스: ko / en / other

### 분류 결과
```
precision    recall  f1-score   support

          en     0.9646    0.9812    0.9729      4000
          ko     0.8962    0.9916    0.9415      3560
       other     0.9859    0.8755    0.9274      4000

    accuracy                         0.9478     11560
   macro avg     0.9489    0.9494    0.9473     11560
weighted avg     0.9509    0.9478    0.9475     11560
```
### 혼동 행렬 (ko, en, other)
```
[[3530,   29,    1],
 [  26, 3925,   49],
 [ 383,  115, 3502]]
```
### 주요 예측 토큰
- ko: 너무, 맛있어요, 좋아요, 맛은, 정말, 맛있는, 맛이, 맛, 맛있고, 맛있음, 좀, 좋은, 그냥, 최고, 맛도
- en: and, the, great, food, to, always, place, was, for, of, service, delicious, best, is, good
- other: muy, de, comida, el, excelente, buen, lugar, los, bien, un, servicio, bueno, delicioso, rico, en

### 샘플 예측 오류 예시
```
true=en, pred=other, text=Loco salad 😆
true=other, pred=ko, text=Wienie.
true=other, pred=en, text=Awesome bugers!
true=ko, pred=en, text=고기가 정말 최고에요. 밑반찬도 고급스럽고. 고추 양파 무절임은 지구 최고!
This is where the best Korean bbq is at.
true=en, pred=other, text=Nice smoothie
```
