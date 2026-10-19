import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
"""
발표용 급행 ABSA 근사 파이프라인 (2시간 데드라인용, 골드셋 학습 아님을 명시할 것).

방법: 사전학습 다국어 감성분류기(cardiffnlp/twitter-xlm-roberta-base-sentiment)를
      aspect 키워드로 매칭된 문장에 적용 → 리뷰x측면 감성점수 → 장소 고정효과 회귀.

대상: place_stats_th03.csv tier=='test' (ko_n>=30) 115개 장소, ko+en 리뷰 전량.

출력: data/processed/wp1b_aspect_sentence_scores.csv (문장 단위 원자료)
      data/processed/wp1b_aspect_regression.csv (측면별 β/CI/Cohen's d)
      outputs/wp1b_aspect_gap_bar.png
      outputs/wp1b_lang_rating_hist.png
"""

import os
import re
import time

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothicExtraBold.ttf"]:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
import statsmodels.formula.api as smf
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ.setdefault("USE_TF", "0")

REVIEWS_PATH = "data/processed/reviews_final_th03.parquet"
PLACE_PATH = "data/processed/place_stats_th03.csv"
OUT_DIR = "data/processed"
FIG_DIR = "outputs"
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
BATCH_SIZE = 32

ASPECTS = {
    "taste": {
        "ko": ["맛있", "맛없", "음식", "요리", "메뉴", "양념", "국물", "신선", "고소", "달다", "짜다", "매콤", "느끼"],
        "en": ["tasty", "flavor", "delicious", "food", "dish", "fresh", "seasoning", "spicy", "bland", "taste"],
    },
    "service": {
        "ko": ["서비스", "직원", "친절", "웨이팅", "대기", "종업원", "응대", "사장님", "불친절"],
        "en": ["service", "staff", "friendly", "wait", "waiter", "waitress", "server", "rude", "attentive"],
    },
    "value": {
        "ko": ["가격", "가성비", "비싸", "저렴", "값"],
        "en": ["price", "value", "expensive", "cheap", "worth", "pricey", "overpriced", "affordable"],
    },
    "atmosphere": {
        "ko": ["분위기", "인테리어", "자리", "공간", "깨끗", "위생", "좌석"],
        "en": ["atmosphere", "ambiance", "ambience", "decor", "vibe", "clean", "cozy", "seating", "interior"],
    },
}

SENT_SPLIT_RE = re.compile(r"[.!?\n]+")


def log(msg):
    print(f"[wp1b {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data():
    place = pd.read_csv(PLACE_PATH)
    test_ids = set(place.loc[place.tier == "test", "gmap_id"])
    log(f"test-tier 장소: {len(test_ids)}곳")

    df = pd.read_parquet(REVIEWS_PATH, columns=["gmap_id", "text_orig", "lang_group", "is_restaurant", "rating"])
    df = df[df["is_restaurant"] & df["gmap_id"].isin(test_ids) & df["lang_group"].isin(["ko", "en"])].copy()
    df["rating"] = df["rating"].astype(float)
    df["is_ko"] = (df["lang_group"] == "ko").astype(int)
    df = df.reset_index(drop=True)
    df["review_id"] = df.index
    log(f"대상 리뷰: {len(df):,}건 (ko={df.is_ko.sum():,}, en={(1-df.is_ko).sum():,})")
    return df


def split_sentences(text):
    if not isinstance(text, str):
        return []
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if len(s.strip()) >= 2]


def match_aspects(sentence):
    low = sentence.lower()
    hits = []
    for aspect, kws in ASPECTS.items():
        for kw in kws["ko"]:
            if kw in sentence:
                hits.append(aspect)
                break
        else:
            for kw in kws["en"]:
                if kw in low:
                    hits.append(aspect)
                    break
    return hits


def build_sentence_table(df):
    log("문장 분리 + aspect 키워드 매칭 시작")
    rows = []
    for rid, gid, is_ko, text in zip(df.review_id, df.gmap_id, df.is_ko, df.text_orig):
        for sent in split_sentences(text):
            aspects = match_aspects(sent)
            for a in aspects:
                rows.append((rid, gid, is_ko, a, sent))
    sent_df = pd.DataFrame(rows, columns=["review_id", "gmap_id", "is_ko", "aspect", "sentence"])
    log(f"매칭된 (리뷰x측면x문장) 행: {len(sent_df):,}건")
    print(sent_df.groupby("aspect").size())
    return sent_df


def score_sentences(sent_df):
    log(f"감성모델 로드: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    sentences = sent_df["sentence"].tolist()
    n = len(sentences)
    scores = np.zeros(n, dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            batch = sentences[i:i + BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=64)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).numpy()  # [neg, neu, pos]
            scores[i:i + BATCH_SIZE] = probs[:, 2] - probs[:, 0]
            if (i // BATCH_SIZE) % 200 == 0:
                elapsed = time.time() - t0
                log(f"  {i + len(batch):,}/{n:,} 문장 처리 ({elapsed:.1f}s)")

    sent_df = sent_df.copy()
    sent_df["sentiment_score"] = scores
    log(f"전체 추론 완료: {n:,}문장, {time.time()-t0:.1f}s")
    return sent_df


def aggregate_review_aspect(sent_df, df):
    agg = sent_df.groupby(["review_id", "gmap_id", "is_ko", "aspect"])["sentiment_score"].mean().reset_index()
    return agg


def run_regressions(agg):
    results = []
    for aspect in ASPECTS:
        sub = agg[agg.aspect == aspect].copy()
        sub["gmap_id"] = sub["gmap_id"].astype(str)
        n_places = sub["gmap_id"].nunique()
        if len(sub) < 30 or n_places < 2:
            continue
        model = smf.ols("sentiment_score ~ is_ko + C(gmap_id)", data=sub).fit()
        beta = model.params["is_ko"]
        ci_low, ci_high = model.conf_int().loc["is_ko"]
        pval = model.pvalues["is_ko"]
        resid_sd = model.resid.std()
        cohens_d = beta / resid_sd if resid_sd > 0 else np.nan
        results.append({
            "aspect": aspect, "n_obs": len(sub), "n_places": n_places,
            "n_ko": int(sub.is_ko.sum()), "n_en": int((1 - sub.is_ko).sum()),
            "beta_is_ko": beta, "ci_low": ci_low, "ci_high": ci_high,
            "p_value": pval, "cohens_d": cohens_d,
        })
    return pd.DataFrame(results)


def make_bar_chart(reg_df):
    if reg_df.empty:
        log("회귀 결과 없음 — 차트 생략")
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = reg_df.sort_values("beta_is_ko")
    yerr = [order.beta_is_ko - order.ci_low, order.ci_high - order.beta_is_ko]
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in order.beta_is_ko]
    ax.barh(order.aspect, order.beta_is_ko, xerr=yerr, color=colors, capsize=4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("β (한국어 리뷰 - 영어 리뷰, 장소 고정효과 통제 후 감성점수 차이)")
    ax.set_title("측면별 언어 간 감성 격차 (test-tier 115곳)")
    plt.tight_layout()
    path = f"{FIG_DIR}/wp1b_aspect_gap_bar.png"
    plt.savefig(path, dpi=150)
    log(f"저장: {path}")


def make_rating_hist(df):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(0.5, 6.5, 1)
    for lang, color in [("ko", "#d62728"), ("en", "#1f77b4")]:
        vals = df.loc[df.lang_group == lang, "rating"].dropna()
        ax.hist(vals, bins=bins, density=True, alpha=0.55, label=f"{lang} (n={len(vals):,})", color=color)
    ax.set_xlabel("평점")
    ax.set_ylabel("비율")
    ax.set_title("언어별 평점 분포 (test-tier 115곳)")
    ax.legend()
    plt.tight_layout()
    path = f"{FIG_DIR}/wp1b_lang_rating_hist.png"
    plt.savefig(path, dpi=150)
    log(f"저장: {path}")


def overall_rating_regression(df):
    log("전체 평점 장소 고정효과 회귀 (평점 변환식 근거)")
    sub = df.dropna(subset=["rating"]).copy()
    sub["gmap_id"] = sub["gmap_id"].astype(str)
    model = smf.ols("rating ~ is_ko + C(gmap_id)", data=sub).fit()
    beta = model.params["is_ko"]
    ci_low, ci_high = model.conf_int().loc["is_ko"]
    pval = model.pvalues["is_ko"]
    resid_sd = model.resid.std()
    cohens_d = beta / resid_sd if resid_sd > 0 else np.nan
    print()
    print("=== 전체 평점 회귀: rating ~ is_ko + C(gmap_id) ===")
    print(f"β(한국어 효과) = {beta:.4f}  95% CI [{ci_low:.4f}, {ci_high:.4f}]  p={pval:.3g}  Cohen's d={cohens_d:.4f}")
    return {"beta": beta, "ci_low": ci_low, "ci_high": ci_high, "p_value": pval, "cohens_d": cohens_d}


def main():
    df = load_data()
    make_rating_hist(df)

    sent_df = build_sentence_table(df)
    sent_df = score_sentences(sent_df)
    sent_df.to_csv(f"{OUT_DIR}/wp1b_aspect_sentence_scores.csv", index=False, encoding="utf-8-sig")

    agg = aggregate_review_aspect(sent_df, df)
    reg_df = run_regressions(agg)
    reg_df.to_csv(f"{OUT_DIR}/wp1b_aspect_regression.csv", index=False, encoding="utf-8-sig")

    print()
    print("=" * 70)
    print("측면별 장소 고정효과 회귀 결과 (sentiment_score ~ is_ko + C(gmap_id))")
    print("=" * 70)
    print(reg_df.to_string(index=False))

    make_bar_chart(reg_df)
    overall_rating_regression(df)

    log("wp1b 완료")


if __name__ == "__main__":
    main()
