"""
문화별 리뷰 인식 차이 분석 — 3단계 최종 데이터셋 빌드 파이프라인.

data/raw/share/final_reviews_v21.jsonl (리뷰) +
data/raw/share/la_restaurants_meta_v21.jsonl (장소 메타)
  -> data/processed/reviews_final_th03.parquet
     data/processed/place_stats_th03.csv
     data/processed/category_stats_th03.csv

data/raw/ 는 절대 건드리지 않는다 (읽기 전용).

실행:
    python3 src/07_build_dataset.py
"""

import json
import os
import re
import sys
import time
import unicodedata

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

# ============================================================
# 튜닝 상수
# ============================================================
KO_THRESHOLD = 0.3   # hangul_ratio_v2 >= 이 값이면 한국어(ko)
MIN_KO_TEST = 30     # 이 이상이면 통계 검정 대상 (tier='test')
MIN_KO_DISPLAY = 10  # 이 이상이면 화면 표시 대상 (tier='display')
LATIN_RATIO = 0.6    # 짧은 영어(en) 판정 기준: 라틴 문자 비율 하한

REVIEW_PATH = "data/raw/share/final_reviews_v21.jsonl"
META_PATH = "data/raw/share/la_restaurants_meta_v21.jsonl"
OUT_DIR = "data/processed"
REVIEWS_OUT = os.path.join(OUT_DIR, "reviews_final_th03.parquet")
PLACE_OUT = os.path.join(OUT_DIR, "place_stats_th03.csv")
CATEGORY_OUT = os.path.join(OUT_DIR, "category_stats_th03.csv")

CHUNKSIZE = 300_000
ALPHA = 0.05

FOOD_KEYWORDS = [
    "restaurant", "cafe", "coffee", "bar", "bakery", "deli", "diner", "pizza", "sushi", "bbq",
    "grill", "steak", "buffet", "noodle", "ramen", "taco", "burger", "pub", "brewery",
    "dessert", "ice cream", "sandwich", "brunch", "bistro", "food",
    "donut", "yogurt", "acai", "tea", "juice", "cookie",
    # 제외된 업종 847곳 전수 점검(누락 키워드 진단)에서 확인된 실제 음식류 업종 보강.
    # 'ham'은 3글자라 오탐 위험이 커서 의도적으로 제외.
    "bagel", "salad", "cake", "pastry", "chicken", "pie",
    "tamale", "kebab", "beer", "crab", "crep",
]

# 'bar' -> 'Barber shop'/'Eyebrow bar', 'pub' -> 'Public golf course' 처럼
# 짧은 키워드가 비식당 카테고리를 부분일치로 통과시키는 것 확인된 항목들.
# check_bar_keyword_conflict()가 매 실행마다 이 목록의 근거를 재확인해서 출력한다.
EXCLUDE_CATEGORIES = ["barber shop", "eyebrow bar", "public golf course"]

RE_ORIGINAL = re.compile(r"\(original\)\s*", re.IGNORECASE)
RE_HANGUL_SYLLABLE = re.compile(r"[가-힣]")
RE_LATIN = re.compile(r"[a-zA-Z]")
HANGUL_LO, HANGUL_HI = 0xAC00, 0xD7A3


def normalize_text(s):
    """악센트/발음기호 제거 + 소문자화. 'Açaí shop' -> 'acai shop', 'Crêperie' -> 'creperie'.

    NFD로 분해한 뒤 결합문자(유니코드 카테고리 'Mn', 예: 악센트 기호)를 제거한다.
    카테고리 문자열과 키워드 양쪽에 동일하게 적용해야 'acai'/'crep' 같은 짧은 라틴 키워드가
    발음기호 포함 카테고리와 매칭될 수 있다.
    """
    if s is None:
        return ""
    nfd = unicodedata.normalize("NFD", s.lower())
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


FOOD_PATTERN = re.compile("|".join(re.escape(normalize_text(k)) for k in FOOD_KEYWORDS))
EXCLUDE_PATTERN = re.compile("|".join(re.escape(normalize_text(k)) for k in EXCLUDE_CATEGORIES))

_OLD_FOOD_KEYWORDS = [
    "restaurant", "cafe", "coffee", "bar", "bakery", "deli", "diner", "pizza", "sushi", "bbq",
    "grill", "steak", "buffet", "noodle", "ramen", "taco", "burger", "pub", "brewery",
    "dessert", "ice cream", "sandwich", "brunch", "bistro", "food",
    "donut", "yogurt", "acai", "tea", "juice", "cookie",
]
_OLD_FOOD_PATTERN = re.compile("|".join(re.escape(normalize_text(k)) for k in _OLD_FOOD_KEYWORDS))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hangul_ratio_v2_scalar(text):
    """한글 음절(가-힣) 비율. 분모는 공백·문장부호 제외 전체 문자."""
    if not text:
        return 0.0
    hangul_n = 0
    denom_n = 0
    for ch in text:
        cp = ord(ch)
        if HANGUL_LO <= cp <= HANGUL_HI:
            hangul_n += 1
        if ch.isspace():
            continue
        if unicodedata.category(ch).startswith("P"):
            continue
        denom_n += 1
    if denom_n == 0:
        return 0.0
    return hangul_n / denom_n


# ============================================================
# 0. 메타 로드 + 업종 필터 진단
# ============================================================
def load_meta():
    log(f"메타 로드: {META_PATH}")
    meta = pd.read_json(META_PATH, lines=True, dtype={"gmap_id": str})
    meta = meta[["gmap_id", "name", "address", "latitude", "longitude", "_primary_category"]].copy()
    log(f"메타 {len(meta):,}행 로드 완료")
    return meta


def check_bar_keyword_conflict(meta):
    """'bar' 등 짧은 키워드가 비식당 카테고리를 잘못 통과시키는지 실측 확인."""
    cats = meta["_primary_category"].dropna().unique()
    counts = meta["_primary_category"].value_counts()
    norm_cats = {c: normalize_text(c) for c in cats}

    print()
    print("=" * 70)
    print("업종 필터 키워드 충돌 점검 ('bar' -> 'Barber shop' 등)")
    print("=" * 70)
    bar_hits = sorted(c for c in cats if "bar" in norm_cats[c])
    print(f"\n'bar' 키워드로 매칭되는 _primary_category ({len(bar_hits)}개):")
    exclude_norm_set_early = {normalize_text(e) for e in EXCLUDE_CATEGORIES}
    for c in bar_hits:
        mark = "  <- EXCLUDE_CATEGORIES 처리됨" if norm_cats[c] in exclude_norm_set_early else ""
        print(f"    {counts.get(c, 0):>5d}  {c!r}{mark}")

    pub_hits = sorted(c for c in cats if "pub" in norm_cats[c])
    print(f"\n'pub' 키워드로 매칭되는 _primary_category ({len(pub_hits)}개):")
    for c in pub_hits:
        mark = "  <- EXCLUDE_CATEGORIES 처리됨" if norm_cats[c] in exclude_norm_set_early else ""
        print(f"    {counts.get(c, 0):>5d}  {c!r}{mark}")

    exclude_norm_set = {normalize_text(e) for e in EXCLUDE_CATEGORIES}
    all_food_matches = sorted(c for c in cats if FOOD_PATTERN.search(norm_cats[c]))
    unflagged_short = [c for c in all_food_matches if norm_cats[c] not in exclude_norm_set]
    print(f"\nFOOD_KEYWORDS 매칭 카테고리 총 {len(all_food_matches)}개 "
          f"(그 중 EXCLUDE_CATEGORIES 처리 {len(all_food_matches) - len(unflagged_short)}개)")
    print()

    print("=" * 70)
    print("악센트 정규화 확인 (NFD + 결합문자 제거)")
    print("=" * 70)
    for target_cat, keyword in [("Açaí shop", "acai"), ("Crêperie", "crep")]:
        norm = normalize_text(target_cat)
        matched_before = keyword in target_cat.lower()
        matched_after = keyword in norm
        present = target_cat in cats
        print(f"  {target_cat!r} (메타에 존재: {present}) -> 정규화: {norm!r}")
        print(f"    정규화 전 '{keyword}' 매칭: {matched_before}  /  정규화 후 매칭: {matched_after}")
    print()


def compute_is_restaurant(meta):
    cat_norm = meta["_primary_category"].fillna("").apply(normalize_text)
    is_excluded = cat_norm.str.contains(EXCLUDE_PATTERN, regex=True)
    is_food = cat_norm.str.contains(FOOD_PATTERN, regex=True)
    is_restaurant = is_food & ~is_excluded
    n_true = int(is_restaurant.sum())
    log(f"is_restaurant 계산 완료: {n_true:,} / {len(meta):,}곳이 식당으로 분류됨 "
        f"(제외키워드로 걸러진 곳: {int((is_food & is_excluded).sum())}곳)")
    return is_restaurant


def report_keyword_expansion_effect(meta):
    """새로 추가된 키워드(bagel/salad/cake/...)로 새로 포함된 장소를 보여준다.
    meta는 compute_is_restaurant() 실행 후, 즉 'is_restaurant' 컬럼이 이미 채워진 상태여야 한다.
    """
    cat_norm = meta["_primary_category"].fillna("").apply(normalize_text)
    is_excluded = cat_norm.str.contains(EXCLUDE_PATTERN, regex=True)
    old_is_restaurant = cat_norm.str.contains(_OLD_FOOD_PATTERN, regex=True) & ~is_excluded

    newly_included = meta[meta["is_restaurant"] & ~old_is_restaurant]
    print()
    print("=" * 70)
    print("FOOD_KEYWORDS 보강 효과")
    print("=" * 70)
    print(f"보강 전 is_restaurant=True: {int(old_is_restaurant.sum()):,}곳")
    print(f"보강 후 is_restaurant=True: {int(meta['is_restaurant'].sum()):,}곳 "
          f"(+{len(newly_included):,}곳)")
    print()
    print("새로 포함된 카테고리 (장소 수 기준):")
    print(newly_included["_primary_category"].value_counts().to_string())
    print()


# ============================================================
# 1~4. 리뷰 스트리밍 처리 (파싱 -> 비율 -> lang_group -> is_restaurant)
# ============================================================
def process_chunk(chunk, id_to_cat, id_to_is_restaurant):
    keep_cols = ["user_id", "name", "time", "rating", "text", "pics", "resp",
                 "gmap_id", "lang_final", "lang_method", "hangul_ratio", "langdetect_raw"]
    df = chunk[keep_cols].copy()

    df["rating"] = df["rating"].astype("Int64")

    df["pics"] = df["pics"].apply(lambda v: json.dumps(v) if v is not None else None)
    df["resp"] = df["resp"].apply(lambda v: json.dumps(v) if v is not None else None)

    # ---- 2. 번역 마커 파싱 ----
    text_col = df["text"]
    null_text_mask = text_col.isna()
    origs, has_trans = [], []
    for t, is_null in zip(text_col, null_text_mask):
        if is_null:
            origs.append(None)
            has_trans.append(False)
            continue
        m = RE_ORIGINAL.search(t)
        if m:
            origs.append(t[m.end():])
            has_trans.append(True)
        else:
            origs.append(t)
            has_trans.append(False)
    df["text_orig"] = origs
    df["has_translation"] = pd.array(has_trans, dtype="bool")

    # ---- 3. hangul_ratio_v2 ----
    df["hangul_ratio_v2"] = df["text_orig"].apply(hangul_ratio_v2_scalar)

    # ---- 4. lang_group ----
    text_orig_filled = df["text_orig"].fillna("")
    char_len = text_orig_filled.str.len()
    has_hangul = text_orig_filled.str.contains(RE_HANGUL_SYLLABLE, regex=True)
    latin_ratio = text_orig_filled.str.count(RE_LATIN) / char_len.replace(0, np.nan)

    is_ko = df["hangul_ratio_v2"] >= KO_THRESHOLD
    en_by_lang_final = df["lang_final"] == "en"
    en_by_short_latin = (~has_hangul) & (latin_ratio >= LATIN_RATIO) & (char_len >= 2)
    is_en = (~is_ko) & (df["has_translation"] == False) & (en_by_lang_final | en_by_short_latin)

    df["lang_group"] = np.select([is_ko, is_en], ["ko", "en"], default="other")

    # ---- 5. is_restaurant (메타 조인) ----
    df["is_restaurant"] = df["gmap_id"].map(id_to_is_restaurant).fillna(False).astype(bool)

    return df


def build_reviews_parquet(id_to_cat, id_to_is_restaurant):
    os.makedirs(OUT_DIR, exist_ok=True)
    log(f"리뷰 스트리밍 시작: {REVIEW_PATH} (chunksize={CHUNKSIZE:,})")

    t0 = time.time()
    total = 0
    writer = None
    lang_group_counts = {"ko": 0, "en": 0, "other": 0}
    null_rating_count = 0

    reader = pd.read_json(
        REVIEW_PATH, lines=True, chunksize=CHUNKSIZE,
        dtype={"user_id": str, "gmap_id": str},
    )
    for i, chunk in enumerate(reader):
        processed = process_chunk(chunk, id_to_cat, id_to_is_restaurant)
        total += len(processed)

        for g, c in processed["lang_group"].value_counts().items():
            lang_group_counts[g] += int(c)
        null_rating_count += int(processed["rating"].isna().sum())

        table = pa.Table.from_pandas(processed, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(REVIEWS_OUT, table.schema)
        writer.write_table(table)

        log(f"  chunk {i + 1}: {total:,}행 누적 ({time.time() - t0:.1f}s)")

    if writer is not None:
        writer.close()

    log(f"리뷰 저장 완료: {REVIEWS_OUT} ({total:,}행, {time.time() - t0:.1f}s)")
    log(f"  lang_group (스트리밍 집계): {lang_group_counts}")
    log(f"  rating null: {null_rating_count:,}건")
    return total, lang_group_counts


# ============================================================
# 6. 장소 단위 / 업종 단위 통계 산출
# ============================================================
def compute_place_stats(meta):
    log("장소 단위 통계 계산 시작")
    t0 = time.time()

    df = pd.read_parquet(REVIEWS_OUT, columns=["gmap_id", "rating", "lang_group", "is_restaurant"])
    df = df[df["is_restaurant"]].copy()
    df["rating"] = df["rating"].astype(float)
    rated = df[df["rating"].notna()]

    ko_counts = df[df["lang_group"] == "ko"].groupby("gmap_id").size()
    en_counts = df[df["lang_group"] == "en"].groupby("gmap_id").size()
    ko_mean = rated[rated["lang_group"] == "ko"].groupby("gmap_id")["rating"].mean()
    en_mean = rated[rated["lang_group"] == "en"].groupby("gmap_id")["rating"].mean()

    all_places = sorted(set(ko_counts.index) | set(en_counts.index))
    place_df = pd.DataFrame({"gmap_id": all_places})
    place_df["ko_n"] = place_df["gmap_id"].map(ko_counts).fillna(0).astype(int)
    place_df["en_n"] = place_df["gmap_id"].map(en_counts).fillna(0).astype(int)
    place_df["ko_mean"] = place_df["gmap_id"].map(ko_mean)
    place_df["en_mean"] = place_df["gmap_id"].map(en_mean)
    place_df["gap"] = place_df["ko_mean"] - place_df["en_mean"]

    place_df["tier"] = np.select(
        [place_df["ko_n"] >= MIN_KO_TEST, place_df["ko_n"] >= MIN_KO_DISPLAY],
        ["test", "display"],
        default="fallback",
    )

    test_places = place_df[place_df["ko_n"] >= MIN_KO_TEST]["gmap_id"].tolist()
    target_set = set(test_places)
    rated_sub = rated[rated["gmap_id"].isin(target_set)]
    ko_arrays = rated_sub[rated_sub["lang_group"] == "ko"].groupby("gmap_id")["rating"].apply(np.asarray)
    en_arrays = rated_sub[rated_sub["lang_group"] == "en"].groupby("gmap_id")["rating"].apply(np.asarray)

    p_map = {}
    for gid in test_places:
        ko_vals = ko_arrays.get(gid, np.array([]))
        en_vals = en_arrays.get(gid, np.array([]))
        if len(ko_vals) < 2 or len(en_vals) < 2:
            continue
        if ko_vals.var(ddof=1) == 0 or en_vals.var(ddof=1) == 0:
            continue
        _, pval = stats.ttest_ind(ko_vals, en_vals, equal_var=False)
        p_map[gid] = pval

    place_df["p_value"] = place_df["gmap_id"].map(p_map)

    valid = place_df["p_value"].notna()
    p = place_df.loc[valid, "p_value"].to_numpy()
    n = len(p)
    if n > 0:
        order = np.argsort(p)
        ranked_p = p[order]
        bh_adj = ranked_p * n / np.arange(1, n + 1)
        bh_adj = np.minimum.accumulate(bh_adj[::-1])[::-1]
        bh_adj = np.clip(bh_adj, 0, 1)
        q = np.empty(n)
        q[order] = bh_adj
        place_df.loc[valid, "q_value"] = q
    else:
        place_df["q_value"] = np.nan

    place_df["is_significant"] = place_df["q_value"] < ALPHA

    place_df = place_df.merge(
        meta.rename(columns={"latitude": "lat", "longitude": "lng", "_primary_category": "primary_category"}),
        on="gmap_id", how="left",
    )
    place_df = place_df[[
        "gmap_id", "name", "primary_category", "address", "lat", "lng",
        "ko_n", "en_n", "ko_mean", "en_mean", "gap",
        "p_value", "q_value", "is_significant", "tier",
    ]]

    place_df.to_csv(PLACE_OUT, index=False, encoding="utf-8-sig")
    log(f"장소 통계 저장 완료: {PLACE_OUT} ({len(place_df):,}곳, {time.time() - t0:.1f}s)")
    return place_df


def compute_category_stats(meta):
    log("업종 단위 통계 계산 시작")
    t0 = time.time()

    df = pd.read_parquet(REVIEWS_OUT, columns=["gmap_id", "rating", "lang_group", "is_restaurant"])
    df = df[df["is_restaurant"]].copy()
    df["rating"] = df["rating"].astype(float)

    cat_map = meta.set_index("gmap_id")["_primary_category"]
    df["primary_category"] = df["gmap_id"].map(cat_map)

    rated = df[df["rating"].notna()]
    baseline_ko = rated.loc[rated["lang_group"] == "ko", "rating"].mean()
    baseline_en = rated.loc[rated["lang_group"] == "en", "rating"].mean()
    baseline_gap = baseline_ko - baseline_en
    log(f"  전체 기준점 (식당만): ko={baseline_ko:.4f} en={baseline_en:.4f} gap={baseline_gap:.4f}")

    place_n = df.groupby("primary_category")["gmap_id"].nunique()
    ko_n = df[df["lang_group"] == "ko"].groupby("primary_category").size()
    en_n = df[df["lang_group"] == "en"].groupby("primary_category").size()
    ko_mean = rated[rated["lang_group"] == "ko"].groupby("primary_category")["rating"].mean()
    en_mean = rated[rated["lang_group"] == "en"].groupby("primary_category")["rating"].mean()

    cat_df = pd.DataFrame({"place_n": place_n}).reset_index()
    cat_df["ko_n"] = cat_df["primary_category"].map(ko_n).fillna(0).astype(int)
    cat_df["en_n"] = cat_df["primary_category"].map(en_n).fillna(0).astype(int)
    cat_df["ko_mean"] = cat_df["primary_category"].map(ko_mean)
    cat_df["en_mean"] = cat_df["primary_category"].map(en_mean)
    cat_df["abs_gap"] = cat_df["ko_mean"] - cat_df["en_mean"]
    cat_df["rel_gap"] = cat_df["abs_gap"] - baseline_gap

    ko_reviews = df[df["lang_group"] == "ko"]
    ko_place_n = ko_reviews.groupby("primary_category")["gmap_id"].nunique()
    ko_per_place_cat = ko_reviews.groupby(["primary_category", "gmap_id"]).size()
    top_place_ko = ko_per_place_cat.groupby(level=0).max()
    total_ko_by_cat = ko_per_place_cat.groupby(level=0).sum()
    top_place_ko_share = top_place_ko / total_ko_by_cat

    cat_df["ko_place_n"] = cat_df["primary_category"].map(ko_place_n).fillna(0).astype(int)
    cat_df["top_place_ko_share"] = cat_df["primary_category"].map(top_place_ko_share)

    ko_arrays = rated[rated["lang_group"] == "ko"].groupby("primary_category")["rating"].apply(np.asarray)
    en_arrays = rated[rated["lang_group"] == "en"].groupby("primary_category")["rating"].apply(np.asarray)

    def pval_for(cat, ko_n_val):
        if ko_n_val < MIN_KO_TEST:
            return np.nan
        ko_vals = ko_arrays.get(cat, np.array([]))
        en_vals = en_arrays.get(cat, np.array([]))
        if len(ko_vals) < 2 or len(en_vals) < 2:
            return np.nan
        if ko_vals.var(ddof=1) == 0 or en_vals.var(ddof=1) == 0:
            return np.nan
        return stats.ttest_ind(ko_vals, en_vals, equal_var=False)[1]

    cat_df["p_value"] = [pval_for(c, n) for c, n in zip(cat_df["primary_category"], cat_df["ko_n"])]

    cat_df["is_reportable"] = (
        (cat_df["ko_n"] >= MIN_KO_TEST)
        & (cat_df["ko_place_n"] >= 5)
        & (cat_df["top_place_ko_share"] <= 0.5)
    )

    cat_df["q_value"] = np.nan
    fdr_mask = cat_df["is_reportable"] & cat_df["p_value"].notna()
    p = cat_df.loc[fdr_mask, "p_value"].to_numpy()
    n = len(p)
    if n > 0:
        order = np.argsort(p)
        ranked_p = p[order]
        bh_adj = ranked_p * n / np.arange(1, n + 1)
        bh_adj = np.minimum.accumulate(bh_adj[::-1])[::-1]
        bh_adj = np.clip(bh_adj, 0, 1)
        q = np.empty(n)
        q[order] = bh_adj
        cat_df.loc[fdr_mask, "q_value"] = q
    cat_df["is_significant_fdr"] = cat_df["q_value"] < ALPHA

    cat_df = cat_df.sort_values("rel_gap")

    cat_df.to_csv(CATEGORY_OUT, index=False, encoding="utf-8-sig")
    log(f"업종 통계 저장 완료: {CATEGORY_OUT} ({len(cat_df):,}개 업종, {time.time() - t0:.1f}s)")
    return cat_df, baseline_ko, baseline_en, baseline_gap


# ============================================================
# 검증
# ============================================================
def verify_pre_filter(lang_group_counts):
    print()
    print("=" * 70)
    print("A-1. 업종 필터 적용 '전' 검증 (기존 재확인, 저장된 parquet을 다시 읽어서 확인)")
    print("=" * 70)

    df = pd.read_parquet(REVIEWS_OUT, columns=["lang_group", "rating"])
    counts = df["lang_group"].value_counts().to_dict()
    print(f"lang_group 분포 (재확인): {counts}")

    df["rating"] = df["rating"].astype(float)
    rated = df[df["rating"].notna()]
    ko_mean = rated.loc[rated["lang_group"] == "ko", "rating"].mean()
    en_mean = rated.loc[rated["lang_group"] == "en", "rating"].mean()
    gap = ko_mean - en_mean
    print(f"전체 기준점 (필터 전): ko평균={ko_mean:.4f}  en평균={en_mean:.4f}  격차={gap:.4f}")

    expected_ko, expected_en = 17798, 3246685
    expected_ko_mean, expected_en_mean, expected_gap = 4.0013, 4.2103, -0.2089
    ok = True
    if counts.get("ko", 0) != expected_ko:
        ok = False
        print(f"  [불일치] ko 건수 {counts.get('ko', 0):,} != 기대값 {expected_ko:,}")
    if counts.get("en", 0) != expected_en:
        ok = False
        print(f"  [불일치] en 건수 {counts.get('en', 0):,} != 기대값 {expected_en:,}")
    if abs(ko_mean - expected_ko_mean) > 0.001:
        ok = False
        print(f"  [불일치] ko평균 {ko_mean:.4f} != 기대값 {expected_ko_mean}")
    if abs(en_mean - expected_en_mean) > 0.001:
        ok = False
        print(f"  [불일치] en평균 {en_mean:.4f} != 기대값 {expected_en_mean}")

    if ok:
        print("-> 이전 세션에서 검증된 수치와 일치. 파이프라인 재현 확인됨.")
    else:
        print()
        print("진단: 위 불일치가 발생한 경우 확인할 것 ->")
        print("   1) FOOD_KEYWORDS / EXCLUDE_CATEGORIES 변경 여부")
        print("   2) KO_THRESHOLD, LATIN_RATIO 상수가 0.3 / 0.6 그대로인지")
        print("   3) RE_ORIGINAL 정규식이 대소문자 무시(re.IGNORECASE)로 걸려있는지")
        print("   4) 원본 jsonl(final_reviews_v21.jsonl)이 이전 검증과 동일한 파일인지 (총 3,602,754 확인)")
        print("   5) rating null 70건 처리(Int64 cast)가 chunk마다 동일하게 적용되는지")
    return ok


def verify_post_filter(place_df):
    print()
    print("=" * 70)
    print("A-2. 업종 필터 적용 '후' 검증")
    print("=" * 70)

    place_read = pd.read_csv(PLACE_OUT)
    df = pd.read_parquet(REVIEWS_OUT, columns=["gmap_id", "is_restaurant"])
    n_reviews_restaurant = int(df["is_restaurant"].sum())

    print(f"남은 장소 수 (is_restaurant=True): {len(place_read):,}곳")
    print(f"남은 리뷰 수 (is_restaurant=True): {n_reviews_restaurant:,}건")
    print()
    tier_counts = place_read["tier"].value_counts().to_dict()
    print(f"tier별 장소 수: {tier_counts}")


def verify_reportable(cat_df):
    print()
    print("=" * 70)
    print("B. is_reportable 업종 확인")
    print("=" * 70)

    reportable = cat_df[cat_df["is_reportable"]].sort_values("rel_gap")
    print(f"is_reportable=True 업종: {len(reportable):,}개 / 전체 {len(cat_df):,}개")
    print()
    cols = ["primary_category", "place_n", "ko_n", "ko_place_n", "top_place_ko_share",
            "abs_gap", "rel_gap", "p_value"]
    print(reportable[cols].to_string(index=False))


def verify_final_keyword_gaps(meta):
    print()
    print("=" * 70)
    print("C. 최종 누락 점검 (is_restaurant=False로 남은 장소, ko 건수 기준)")
    print("=" * 70)

    df = pd.read_parquet(REVIEWS_OUT, columns=["gmap_id", "lang_group", "is_restaurant"])
    excluded_ko = df[(df["lang_group"] == "ko") & (~df["is_restaurant"])].copy()
    cat_map = meta.set_index("gmap_id")["_primary_category"]
    excluded_ko["primary_category"] = excluded_ko["gmap_id"].map(cat_map)

    by_cat = excluded_ko.groupby("primary_category").size().sort_values(ascending=False)
    top20 = by_cat.head(20)
    print("제외된(is_restaurant=False) 장소들의 ko 리뷰 수, 업종별 상위 20개:")
    for cat, n in top20.items():
        print(f"    {n:>4d}  {cat}")

    warn = by_cat[by_cat >= 10]
    print()
    if len(warn) > 0:
        print(f"[경고] ko 리뷰 10건 이상인 제외 업종이 {len(warn)}개 있음 -> 키워드 누락 가능성:")
        for cat, n in warn.items():
            print(f"    {n:>4d}  {cat}")
    else:
        print("경고 없음 -> 제외된 업종 중 ko>=10인 곳 없음.")


def verify_category_fdr(cat_df):
    print()
    print("=" * 70)
    print("D. 업종 단위 BH-FDR 보정 (is_reportable=True 대상)")
    print("=" * 70)

    sub = cat_df[cat_df["is_reportable"]].copy()
    print(f"대상 업종: {len(sub):,}개")

    sub["sig_before"] = sub["p_value"] < ALPHA
    sub["sig_after"] = sub["is_significant_fdr"]
    flipped = sub[sub["sig_before"] != sub["sig_after"]]

    print()
    if len(flipped) > 0:
        print(f"보정 전 후 유의성이 뒤집힌 업종: {len(flipped)}개")
        print(flipped[["primary_category", "ko_n", "p_value", "q_value", "sig_before", "sig_after"]]
              .to_string(index=False))
    else:
        print("보정 전 후 유의성이 뒤집힌 업종 없음.")

    print()
    print(f"보정 후 유의(q<0.05) 업종: {int(sub['sig_after'].sum())}개 / {len(sub)}개")


def print_gmap_id_samples(place_df):
    print()
    print("=" * 70)
    print("3. gmap_id 형식 확인 (place_stats 상위 5개)")
    print("=" * 70)
    for v in place_df["gmap_id"].head(5):
        print(f"  {v!r}")


def print_final_summary(place_df, cat_df, baseline_gap):
    print()
    print("=" * 70)
    print("2. 최종 요약")
    print("=" * 70)

    df = pd.read_parquet(REVIEWS_OUT, columns=["lang_group", "is_restaurant"])
    total_reviews = len(df)
    restaurant_reviews = int(df["is_restaurant"].sum())
    lg_counts = df.loc[df["is_restaurant"], "lang_group"].value_counts().to_dict()
    ko_final = lg_counts.get("ko", 0)

    print(f"전체 리뷰 수: {total_reviews:,}")
    print(f"식당 필터 후 리뷰 수: {restaurant_reviews:,}")
    print(f"식당 필터 후 lang_group: {lg_counts}")

    print()
    tier_counts = place_df["tier"].value_counts().to_dict()
    print(f"tier별 장소 수: {tier_counts}")

    print()
    print(f"필터 후 기준점 격차(식당만, ko평균-en평균): {baseline_gap:.4f}")

    reportable_sig = cat_df[cat_df["is_reportable"] & cat_df["is_significant_fdr"]].sort_values("rel_gap")
    print()
    print(f"is_reportable AND q<0.05 인 업종: {len(reportable_sig):,}개")
    for _, r in reportable_sig.iterrows():
        print(f"    {r['primary_category']}  (rel_gap={r['rel_gap']:.3f}, q={r['q_value']:.3g})")

    return {
        "total_reviews": total_reviews,
        "restaurant_reviews": restaurant_reviews,
        "lg_counts": lg_counts,
        "ko_final": ko_final,
        "tier_counts": tier_counts,
        "reportable_sig": reportable_sig,
        "baseline_gap": baseline_gap,
    }


def write_readme(summary, cat_df):
    lg = summary["lg_counts"]
    tier = summary["tier_counts"]
    n_reportable = int(cat_df["is_reportable"].sum())
    readme_path = os.path.join(OUT_DIR, "README.md")

    reportable_sig_lines = "\n".join(
        f"- **{r['primary_category']}** – rel_gap={r['rel_gap']:.3f}, q={r['q_value']:.3g}, ko_n={r['ko_n']}"
        for _, r in summary["reportable_sig"].iterrows()
    ) or "(없음)"

    content = f"""# data/processed — 문화별 리뷰 인식 차이 분석 최종 산출물

`src/07_build_dataset.py`가 `data/raw/share/final_reviews_v21.jsonl` +
`data/raw/share/la_restaurants_meta_v21.jsonl`로부터 생성한 최종 데이터셋.

## 핵심 수치 (최근 실행 기준)

- 전체 리뷰: {summary['total_reviews']:,}건 -> 식당 필터 후: {summary['restaurant_reviews']:,}건
- 식당 필터 후 lang_group: `{lg}`
- tier별 장소 수: `{tier}`
- 필터 후 기준점 격차: {summary['baseline_gap']:.4f}

## is_reportable=True 중 q<0.05

{reportable_sig_lines}
"""

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    log(f"README 저장 완료: {readme_path}")


# ============================================================
# main
# ============================================================
def main():
    t_start = time.time()
    log("=" * 70)
    log("07_build_dataset.py 시작")
    log("=" * 70)

    if not os.path.exists(REVIEW_PATH):
        sys.exit(f"리뷰 원본을 찾을 수 없음: {REVIEW_PATH}")
    if not os.path.exists(META_PATH):
        sys.exit(f"메타 원본을 찾을 수 없음: {META_PATH}")

    meta = load_meta()
    check_bar_keyword_conflict(meta)
    meta["is_restaurant"] = compute_is_restaurant(meta)
    report_keyword_expansion_effect(meta)

    id_to_cat = meta.set_index("gmap_id")["_primary_category"].to_dict()
    id_to_is_restaurant = meta.set_index("gmap_id")["is_restaurant"].to_dict()

    total, lang_group_counts = build_reviews_parquet(id_to_cat, id_to_is_restaurant)

    ok = verify_pre_filter(lang_group_counts)

    place_df = compute_place_stats(meta)
    cat_df, baseline_ko, baseline_en, baseline_gap = compute_category_stats(meta)

    verify_post_filter(place_df)
    verify_reportable(cat_df)
    verify_final_keyword_gaps(meta)
    verify_category_fdr(cat_df)

    summary = print_final_summary(place_df, cat_df, baseline_gap)
    print_gmap_id_samples(place_df)
    write_readme(summary, cat_df)

    print()
    log("=" * 70)
    log(f"07_build_dataset.py 완료 (총 {time.time() - t_start:.1f}s)")
    log(f"  {REVIEWS_OUT}")
    log(f"  {PLACE_OUT}")
    log(f"  {CATEGORY_OUT}")
    log(f"  {os.path.join(OUT_DIR, 'README.md')}")
    log("=" * 70)


if __name__ == "__main__":
    main()
