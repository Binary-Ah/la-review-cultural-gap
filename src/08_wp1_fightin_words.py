"""
WP1 — Fightin' Words 대조 분석 (작업지시서 §5 WP1)

한식당 vs 비한식당 두 조건 각각에서 한국어 리뷰(ko) vs 영어 리뷰(en)의
어휘 로그오즈비(Monroe et al. 2008, informative Dirichlet prior)를 계산하고,
동일 분할에서 평점 분포의 Jensen-Shannon Divergence를 비교한다.

검정 가능한 예측(§4.1③): JSD(ko,en | 한식당) > JSD(ko,en | 비한식당)

입력: data/processed/reviews_final_th03.parquet, data/raw/share/la_restaurants_meta_v21.jsonl
출력: data/processed/wp1_fightin_words_hansikdang.csv
      data/processed/wp1_fightin_words_bihansikdang.csv
      data/processed/wp1_jsd_summary.csv
"""

import re
from collections import Counter

import numpy as np
import pandas as pd
from kiwipiepy import Kiwi
from nltk.corpus import stopwords
from scipy.spatial.distance import jensenshannon

REVIEWS_PATH = "data/processed/reviews_final_th03.parquet"
META_PATH = "data/raw/share/la_restaurants_meta_v21.jsonl"
OUT_DIR = "data/processed"

KO_CONTENT_TAGS = {"NNG", "NNP", "VV", "VA"}
EN_STOPWORDS = set(stopwords.words("english"))
EN_WORD_RE = re.compile(r"[a-zA-Z]+")

Z_SIG = 1.96
TOP_N = 30


def log(msg):
    print(f"[wp1] {msg}", flush=True)


def load_data():
    log(f"리뷰 로드: {REVIEWS_PATH}")
    df = pd.read_parquet(
        REVIEWS_PATH,
        columns=["gmap_id", "text_orig", "lang_group", "is_restaurant", "rating"],
    )
    df = df[df["is_restaurant"] & df["lang_group"].isin(["ko", "en"])].copy()
    df["rating"] = df["rating"].astype(float)

    log(f"메타 로드: {META_PATH}")
    meta = pd.read_json(META_PATH, lines=True, dtype={"gmap_id": str})
    meta = meta[["gmap_id", "_primary_category"]]
    meta["is_korean_category"] = meta["_primary_category"].str.contains(
        "Korean", case=False, na=False
    )

    df = df.merge(meta[["gmap_id", "is_korean_category"]], on="gmap_id", how="left")
    df["split"] = np.where(df["is_korean_category"], "한식당", "비한식당")
    log(f"대상 리뷰: {len(df):,}건 (ko={int((df.lang_group=='ko').sum()):,}, "
        f"en={int((df.lang_group=='en').sum()):,})")
    return df


def tokenize_ko_batch(texts, kiwi):
    """kiwipiepy로 한국어 내용어(명사/동사/형용사) 형태소를 추출한다."""
    counter = Counter()
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    for result in kiwi.analyze(texts, top_n=1):
        tokens = result[0][0]
        for tok in tokens:
            if tok.tag in KO_CONTENT_TAGS:
                form = tok.form
                if tok.tag in ("VV", "VA"):
                    form = form + "다"
                if len(form) >= 1:
                    counter[form] += 1
    return counter


def tokenize_en_bulk(texts):
    """영어 리뷰 텍스트 전체를 한 번에 정규식으로 토큰화 (불용어 제거)."""
    texts = [t for t in texts if isinstance(t, str)]
    blob = "\n".join(texts).lower()
    words = EN_WORD_RE.findall(blob)
    counter = Counter(w for w in words if len(w) >= 2 and w not in EN_STOPWORDS)
    return counter


def monroe_log_odds(counts_i, counts_j, background):
    """Monroe et al. (2008) informative Dirichlet prior log-odds-ratio + z-score.

    counts_i, counts_j: Counter — 비교 대상 두 코퍼스의 단어 빈도
    background: Counter — 사전(prior) 의사빈도로 쓰일 배경 코퍼스 빈도
    """
    vocab = [w for w in background if background[w] > 0 and (w in counts_i or w in counts_j)]
    a0 = sum(background[w] for w in vocab)
    n_i = sum(counts_i.values())
    n_j = sum(counts_j.values())

    rows = []
    for w in vocab:
        a_w = background[w]
        y_i = counts_i.get(w, 0)
        y_j = counts_j.get(w, 0)
        log_odds_i = np.log((y_i + a_w) / (n_i + a0 - y_i - a_w))
        log_odds_j = np.log((y_j + a_w) / (n_j + a0 - y_j - a_w))
        delta = log_odds_i - log_odds_j
        var = 1.0 / (y_i + a_w) + 1.0 / (y_j + a_w)
        z = delta / np.sqrt(var)
        rows.append((w, y_i, y_j, delta, z))

    out = pd.DataFrame(rows, columns=["word", "count_i", "count_j", "log_odds_delta", "z"])
    return out.sort_values("z", ascending=False).reset_index(drop=True)


def run_split(name, sub_df, kiwi):
    log(f"=== {name}: 토큰화 시작 (ko={int((sub_df.lang_group=='ko').sum()):,}건, "
        f"en={int((sub_df.lang_group=='en').sum()):,}건) ===")
    ko_texts = sub_df.loc[sub_df.lang_group == "ko", "text_orig"]
    en_texts = sub_df.loc[sub_df.lang_group == "en", "text_orig"]

    ko_counter = tokenize_ko_batch(ko_texts.tolist(), kiwi)
    log(f"  {name} ko 어휘: {len(ko_counter):,}종, 총 {sum(ko_counter.values()):,}토큰")
    en_counter = tokenize_en_bulk(en_texts.tolist())
    log(f"  {name} en 어휘: {len(en_counter):,}종, 총 {sum(en_counter.values()):,}토큰")

    return ko_counter, en_counter


def main():
    df = load_data()
    kiwi = Kiwi()

    splits = {}
    for name in ["한식당", "비한식당"]:
        sub = df[df["split"] == name]
        splits[name] = run_split(name, sub, kiwi)

    # 전체 리뷰(한식당+비한식당, ko+en)를 배경 prior로 사용 — 두 분할 간 z-score 비교 가능성 확보
    background = Counter()
    for ko_c, en_c in splits.values():
        background.update(ko_c)
        background.update(en_c)
    log(f"전체 배경 어휘: {len(background):,}종, 총 {sum(background.values()):,}토큰")

    results = {}
    for name, (ko_c, en_c) in splits.items():
        log(f"=== {name}: Monroe log-odds (ko vs en) 계산 ===")
        res = monroe_log_odds(ko_c, en_c, background)
        out_path = f"{OUT_DIR}/wp1_fightin_words_{'hansikdang' if name=='한식당' else 'bihansikdang'}.csv"
        res.to_csv(out_path, index=False, encoding="utf-8-sig")
        log(f"  저장: {out_path} ({len(res):,}개 단어)")
        results[name] = res

    for name, res in results.items():
        print()
        print("=" * 70)
        print(f"{name} — 한국어 쪽으로 유의하게 치우친 단어 top {TOP_N} (z > {Z_SIG})")
        print("=" * 70)
        top_ko = res[res.z > Z_SIG].head(TOP_N)
        print(top_ko[["word", "count_i", "count_j", "z"]].to_string(index=False))

        print()
        print(f"{name} — 영어 쪽으로 유의하게 치우친 단어 top {TOP_N} (z < -{Z_SIG})")
        top_en = res[res.z < -Z_SIG].tail(TOP_N).sort_values("z")
        print(top_en[["word", "count_i", "count_j", "z"]].to_string(index=False))

    # ---- JSD: 평점 분포 (ko vs en), 한식당 vs 비한식당 ----
    print()
    print("=" * 70)
    print("JSD(ko,en | split) 비교 — §4.1③ 검정 가능한 예측 확인")
    print("=" * 70)

    jsd_rows = []
    for name in ["한식당", "비한식당"]:
        sub = df[(df["split"] == name) & df["rating"].notna()]
        ko_ratings = sub.loc[sub.lang_group == "ko", "rating"]
        en_ratings = sub.loc[sub.lang_group == "en", "rating"]

        bins = np.arange(1, 6)
        ko_dist = ko_ratings.value_counts(normalize=True).reindex(bins, fill_value=0).sort_index()
        en_dist = en_ratings.value_counts(normalize=True).reindex(bins, fill_value=0).sort_index()

        jsd = jensenshannon(ko_dist.values, en_dist.values, base=2) ** 2  # squared JS distance = JS divergence
        jsd_rows.append({
            "split": name, "ko_n": len(ko_ratings), "en_n": len(en_ratings),
            "ko_mean": ko_ratings.mean(), "en_mean": en_ratings.mean(),
            "jsd": jsd,
        })
        print(f"{name}: ko_n={len(ko_ratings):,} en_n={len(en_ratings):,} "
              f"ko_mean={ko_ratings.mean():.3f} en_mean={en_ratings.mean():.3f} JSD={jsd:.5f}")

    jsd_df = pd.DataFrame(jsd_rows)
    jsd_df.to_csv(f"{OUT_DIR}/wp1_jsd_summary.csv", index=False, encoding="utf-8-sig")

    jsd_han = jsd_df.loc[jsd_df.split == "한식당", "jsd"].iloc[0]
    jsd_bihan = jsd_df.loc[jsd_df.split == "비한식당", "jsd"].iloc[0]
    print()
    print(f"검정 가능한 예측: JSD(한식당)={jsd_han:.5f} > JSD(비한식당)={jsd_bihan:.5f} "
          f"-> {'지지됨 (예측과 일치)' if jsd_han > jsd_bihan else '기각됨 (예측과 불일치)'}")

    print()
    log("WP1 완료")


if __name__ == "__main__":
    main()
