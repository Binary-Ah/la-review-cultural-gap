# LA 음식점 리뷰의 언어별 평점 격차 분석

Google Local 리뷰(캘리포니아)에서 LA 음식점만 골라, **한국어 리뷰와 영어 리뷰가 같은 가게를 다르게 평가하는지**를 분석한 프로젝트.
결과를 바탕으로 사용자 입맛 설문을 리뷰 랭킹에 연결하는 개인화 데모까지 만들었다.

## 규모
- 대상 업체 17,090곳 / 리뷰 3,602,754건 (en 3,246,685 · other 338,271 · ko 17,798)
- 업종 242개 중 보고 가능(reportable) 42개, BH-FDR 보정 후 유의한 업종 21개
- 한국어 리뷰가 많은 업종 예: Korean restaurant (ko 4,873건, ko 평균 3.88 vs en 4.34)

## 파이프라인 (`src/`)

| 단계 | 스크립트 | 내용 |
|---|---|---|
| 00 | `00_power_analysis.py` | "한국어 리뷰 N건 이상" 기준을 검정력 분석으로 역산 |
| 01 | `01_filter_la_places*.py` | 메타데이터 스트리밍 필터: LA 위경도 + 음식점 카테고리. v1 substring 매칭 → v2 정확매칭/제외사전 → v2.1 주 업종(primary) 판정으로 개선 |
| 02 | `02_extract_reviews.py` | 7천만 건 규모 리뷰 파일에서 대상 업체 리뷰만 스트리밍 추출 |
| 03 | `03_language_detect*.py` | langdetect + 한글 비율 규칙으로 언어 판별, OOM 대비 resume 버전 |
| 04 | `04_sample_validation.py` | 업체별 ko/en 리뷰 수 집계, 분석 대상 확정 |
| 05 | `05_embedding_pca.py` | TF-IDF / 다국어 임베딩 + PCA로 언어별 분리 확인 |
| 06–07 | `06_rebuild_final_dataset.py`, `07_build_dataset.py` | 기존 언어분리 결과 재사용하며 최종 parquet + 업체/업종 통계 생성 (tier: test/display/fallback) |
| 08 | `08_wp1_fightin_words.py` | 한식당/비한식당별 ko vs en 어휘 로그오즈비(Fightin' Words) + 평점 분포 JSD |
| 09 | `09_wp1b_absa_quickpass.py`, `09_lang_model_analysis.py` | 맛 속성(매움·기름짐·향) 점수, TF-IDF+LR 언어 분류기 |
| 10–12 | `10_gap_and_text_analysis.py`, `11_demo_visuals.py`, `12_user_personalized_demo.py` | 격차·텍스트 분석 시각화, 설문 벡터 기반 리뷰 랭킹 데모 |

## 결과물 (`outputs/`)
- `ml_analysis_report.md`: 업체/업종별 격차, 언어 분류 모델 평가
- `plots/`: 업종별 맛 속성 Top10, 격차 vs 맛 속성 산점도, 언어별 t-SNE, 언어별 상위 토큰

## 데이터
원본은 UCSD Google Local Data (2021) 캘리포니아 서브셋(`meta-California.json.gz`, `review-California.json.gz`).
용량(수 GB)과 리뷰 작성자 개인정보 때문에 데이터·중간 산출물·리뷰 원문 예시는 레포에 포함하지 않았다.
원본 파일을 `data/raw/`에 두고 `src/` 스크립트를 번호 순서대로 실행하면 재현된다.
