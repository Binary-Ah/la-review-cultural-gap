import pathlib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_data(root):
    place = pd.read_csv(root / "data" / "processed" / "place_stats_th03.csv")
    cat = pd.read_csv(root / "data" / "processed" / "category_stats_th03.csv")
    reviews = pd.read_parquet(root / "data" / "processed" / "reviews_final_th03.parquet", columns=["text_orig", "lang_group"])
    return place, cat, reviews


def summarize(place, cat, reviews):
    lines = []
    lines.append(f"# ML & Data Science 분석 보고서")
    lines.append("")
    lines.append("## 데이터 개요")
    lines.append(f"- 장소 통계 행: {len(place):,}")
    lines.append(f"- 업종 통계 행: {len(cat):,}")
    lines.append(f"- 리뷰 전체 건수: {len(reviews):,}")
    lines.append("- 리뷰 언어 분포:")
    for label, count in reviews["lang_group"].value_counts().items():
        lines.append(f"  - {label}: {count:,}")
    lines.append("")
    lines.append("## place_stats 주요 지표")
    lines.append(f"- tier 분포: {dict(place['tier'].value_counts())}")
    lines.append(f"- 통계적으로 유의미한 장소 수: {int(place['is_significant'].sum())}")
    lines.append("")
    lines.append("## category_stats 주요 지표")
    lines.append(f"- reportable 업종 수: {int(cat['is_reportable'].sum())}")
    lines.append(f"- BH-FDR 유의 업종 수: {int(cat['is_significant_fdr'].sum())}")
    lines.append("")

    lines.append("## 가장 큰 언어 평점 격차 장소")
    lines.append("### ko가 en보다 훨씬 낮은 장소 (Top 10)")
    neg = place.sort_values("gap").head(10)
    lines.extend(format_place_rows(neg))
    lines.append("### ko가 en보다 훨씬 높은 장소 (Top 10)")
    pos = place.sort_values("gap", ascending=False).head(10)
    lines.extend(format_place_rows(pos))
    lines.append("")

    lines.append("## 업종별 상대 격차")
    lines.append("### rel_gap가 가장 작은 업종 (ko가 상대적으로 낮은 업종)")
    rel_neg = cat.sort_values("rel_gap").head(10)
    lines.extend(format_category_rows(rel_neg))
    lines.append("### rel_gap가 가장 큰 업종 (ko가 상대적으로 높은 업종)")
    rel_pos = cat.sort_values("rel_gap", ascending=False).head(10)
    lines.extend(format_category_rows(rel_pos))
    lines.append("")

    lines.append("## 보고 가능 업종 예시")
    lines.append("- Ramen restaurant, Korean restaurant, Asian restaurant, Bakery, Chicken restaurant 등이 reportable로 분류되었습니다.")
    lines.append("")
    top_share = cat[cat['ko_n'] >= 30].assign(ko_rate=lambda df: df['ko_n'] / (df['ko_n'] + df['en_n']))
    top_share = top_share.sort_values('ko_rate', ascending=False).head(10)
    lines.append("## ko 리뷰 비중이 높은 업종")
    lines.extend(format_category_rows(top_share, extra=['ko_rate']))
    lines.append("")
    return "\n".join(lines)


def format_place_rows(df):
    rows = []
    for _, row in df.iterrows():
        rows.append(f"- {row['name']} ({row['primary_category']}): ko_n={int(row['ko_n'])}, en_n={int(row['en_n'])}, ko_mean={row['ko_mean']:.3f}, en_mean={row['en_mean']:.3f}, gap={row['gap']:.3f}, tier={row['tier']}")
    return rows


def format_category_rows(df, extra=None):
    rows = []
    if extra is None:
        extra = []
    for _, row in df.iterrows():
        extra_text = "".join([f", {k}={row[k]:.3f}" for k in extra])
        rows.append(f"- {row['primary_category']}: place_n={int(row['place_n'])}, ko_n={int(row['ko_n'])}, en_n={int(row['en_n'])}, ko_mean={row['ko_mean']:.3f}, en_mean={row['en_mean']:.3f}, abs_gap={row['abs_gap']:.3f}, rel_gap={row['rel_gap']:.3f}{extra_text}")
    return rows


def train_language_model(reviews, max_examples_per_class=20000):
    sample_frames = []
    for label in ['ko', 'en', 'other']:
        group = reviews[reviews['lang_group'] == label]
        if len(group) > max_examples_per_class:
            group = group.sample(max_examples_per_class, random_state=42)
        sample_frames.append(group)
    df = pd.concat(sample_frames).sample(frac=1, random_state=42).reset_index(drop=True)

    X = df['text_orig'].fillna('')
    y = df['lang_group']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vect = TfidfVectorizer(
        min_df=5,
        max_df=0.9,
        ngram_range=(1, 2),
        max_features=25000,
        lowercase=True,
        strip_accents='unicode',
    )
    X_train_tfidf = vect.fit_transform(X_train)
    X_test_tfidf = vect.transform(X_test)

    clf = LogisticRegression(
        solver='saga',
        penalty='l2',
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
    )
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    report = classification_report(y_test, y_pred, digits=4)
    matrix = confusion_matrix(y_test, y_pred, labels=['ko', 'en', 'other'])
    top = get_top_features(vect, clf, labels=['ko', 'en', 'other'], n=15)

    return report, matrix, top, X_test.reset_index(drop=True), y_test.reset_index(drop=True), y_pred, X_train.size


def get_top_features(vect, clf, labels, n=15):
    feature_names = np.array(vect.get_feature_names_out())
    top = {}
    class_index = {label: idx for idx, label in enumerate(clf.classes_)}
    for label in labels:
        idx = class_index[label]
        top_indices = np.argsort(clf.coef_[idx])[-n:][::-1]
        top[label] = feature_names[top_indices].tolist()
    return top


def write_model_summary(report, matrix, top, sample_true, sample_pred, sample_text, root):
    lines = []
    lines.append('## 텍스트 기반 언어 분류 모델')
    lines.append('')
    lines.append('### 모델 요약')
    lines.append('- 모델: TF-IDF + Logistic Regression')
    lines.append('- 클래스: ko / en / other')
    lines.append('')
    lines.append('### 분류 결과')
    lines.append('```')
    lines.append(report.strip())
    lines.append('```')
    lines.append('### 혼동 행렬 (ko, en, other)')
    lines.append('```')
    lines.append(np.array2string(matrix, separator=', '))
    lines.append('```')
    lines.append('### 주요 예측 토큰')
    for label, tokens in top.items():
        lines.append(f'- {label}: {", ".join(tokens[:15])}')
    lines.append('')
    lines.append('### 샘플 예측 오류 예시')
    miss = sample_true[sample_true != sample_pred]
    if len(miss) > 0:
        lines.append('```')
        for idx in miss.index[:5]:
            lines.append(f"true={sample_true[idx]}, pred={sample_pred[idx]}, text={sample_text[idx]}")
        lines.append('```')
    lines.append('')
    return '\n'.join(lines)


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    place, cat, reviews = load_data(root)

    summary_text = summarize(place, cat, reviews)
    report_path = root / 'outputs' / 'ml_analysis_report.md'
    model_report_path = root / 'outputs' / 'ml_analysis_model_report.md'

    report_text = summary_text + '\n'
    report_text += '\n---\n\n'
    report_text += '## 머신러닝 분석 배경 및 의미\n'
    report_text += '\n'.join([
        '- 이 분석은 원본 파이프라인에서 정의한 `lang_group`을 그대로 사용하지만,',
        '  리뷰 텍스트 자체에서 언어 패턴을 학습하는 모델을 만들어 규칙 기반 분류의 보완 가능성을 확인합니다.',
        '- 리뷰가 짧거나 혼합 언어인 경우, 텍스트 기반 모델은 `ko/en/other` 경계 사례를 자동으로 탐지하는 데 도움이 됩니다.',
    ])
    report_text += '\n\n'

    report_text += '## 모델 학습 및 평가 결과\n'
    report, matrix, top, X_test, y_test, y_pred, train_size = train_language_model(reviews)
    model_text = write_model_summary(report, matrix, top, y_test, y_pred, X_test, root)
    report_text += model_text

    report_path.write_text(report_text, encoding='utf-8')
    print(f'Report written to {report_path}')


if __name__ == '__main__':
    main()
