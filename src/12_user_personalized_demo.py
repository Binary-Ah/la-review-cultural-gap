import json
import math
import pathlib
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")

ROOT = pathlib.Path(__file__).resolve().parents[1]
REVIEWS_PATH = ROOT / "data" / "processed" / "reviews_final_th03.parquet"
META_PATH = ROOT / "data" / "processed" / "la_restaurants_meta_v21.jsonl"
OUT_DIR = ROOT / "outputs"
PLOTS_DIR = OUT_DIR / "plots"

LEXICONS = {
    "oily": ["greasy", "oily", "fatty", "rich", "buttery", "heavy", "fried", "cream", "grease", "oil", "fat"],
    "spicy": ["spicy", "hot", "chili", "chilli", "pepper", "heat", "jalapeno", "habanero", "sichuan", "매운", "고추", "칠리"],
    "herb": ["herb", "herby", "cilantro", "coriander", "basil", "rosemary", "thyme", "parsley", "oregano", "mint", "고수", "향신료", "cumin"],
    "exotic": ["authentic", "local", "traditional", "exotic", "unique", "unusual", "foreign", "heritage", "현지", "이국", "전통"],
}

WORD_RE = re.compile(r"[\w\u4e00-\u9fff']+", re.UNICODE)


def normalize_score(x, low=0.0, high=1.0):
    if pd.isna(x):
        return 0.5
    x = float(x)
    if x < 0:
        x = 0
    if x > 1:
        x = 1
    return low + (x * (high - low))


def count_tokens(text, lexicon):
    if not isinstance(text, str):
        return 0
    tokens = WORD_RE.findall(text.lower())
    if not tokens:
        return 0
    score = 0
    for token in lexicon:
        score += sum(1 for t in tokens if token in t)
    return score / max(1, len(tokens))


def build_review_features(reviews_df):
    out = reviews_df.copy()
    out["text_orig"] = out["text_orig"].fillna("")
    for name, lex in LEXICONS.items():
        out[f"{name}_score"] = out["text_orig"].apply(lambda t: count_tokens(t, lex))
    out["rating_norm"] = out["rating"].fillna(3).astype(float) / 5.0
    out["lang_bonus"] = np.where(out["lang_group"] == "ko", 0.12, 0.0)
    return out


def load_data():
    reviews = pd.read_parquet(REVIEWS_PATH, columns=["gmap_id", "text_orig", "lang_group", "rating"])
    reviews = reviews.dropna(subset=["text_orig"]).copy()
    reviews = reviews.sample(n=min(40000, len(reviews)), random_state=42).reset_index(drop=True)
    reviews = build_review_features(reviews)
    return reviews


def user_profile_to_query(profile):
    def as_pref(x):
        return max(1, min(5, int(x)))

    local_pref = as_pref(profile["local_pref"])
    oil_pref = as_pref(profile["oil_pref"])
    spice_pref = as_pref(profile["spice_pref"])
    herb_pref = as_pref(profile["herb_pref"])

    terms = []
    if local_pref >= 4:
        terms.extend(["authentic", "local", "traditional", "unique"])
    elif local_pref <= 2:
        terms.extend(["familiar", "comfort", "classic", "safe"])
    if oil_pref >= 4:
        terms.extend(["rich", "greasy", "buttery", "cream"])
    elif oil_pref <= 2:
        terms.extend(["light", "clean", "less oily", "fresh"])
    if spice_pref >= 4:
        terms.extend(["spicy", "hot", "pepper", "chili"])
    elif spice_pref <= 2:
        terms.extend(["mild", "gentle", "non spicy"])
    if herb_pref >= 4:
        terms.extend(["herb", "cilantro", "cumin", "strong aroma"])
    elif herb_pref <= 2:
        terms.extend(["plain", "subtle", "mild aroma"])
    return terms


def score_review(review_row, profile):
    local_pref = max(1, min(5, int(profile["local_pref"])))
    oil_pref = max(1, min(5, int(profile["oil_pref"])))
    spice_pref = max(1, min(5, int(profile["spice_pref"])))
    herb_pref = max(1, min(5, int(profile["herb_pref"])))

    local_pref_n = (local_pref - 1) / 4.0
    oil_pref_n = (oil_pref - 1) / 4.0
    spice_pref_n = (spice_pref - 1) / 4.0
    herb_pref_n = (herb_pref - 1) / 4.0

    local_score = 1.0 - abs(local_pref_n - normalize_score(review_row["exotic_score"]))
    oil_score = 1.0 - abs(oil_pref_n - normalize_score(review_row["oily_score"]))
    spice_score = 1.0 - abs(spice_pref_n - normalize_score(review_row["spicy_score"]))
    herb_score = 1.0 - abs(herb_pref_n - normalize_score(review_row["herb_score"]))
    rating_score = review_row["rating_norm"]

    # small language prior to reflect cultural matching in the extension experience
    lang_score = review_row["lang_bonus"] if profile.get("prefer_ko_reviews", True) else 0.0

    # text match to query terms (simple lexical overlap)
    query_terms = user_profile_to_query(profile)
    text_terms = [w for w in query_terms if w in review_row["text_orig"].lower()]
    text_score = len(text_terms) / max(1, len(query_terms))

    total = 0.30 * local_score + 0.25 * oil_score + 0.20 * spice_score + 0.15 * herb_score + 0.05 * rating_score + 0.05 * text_score + lang_score
    return total, {
        "local_score": local_score,
        "oil_score": oil_score,
        "spice_score": spice_score,
        "herb_score": herb_score,
        "rating_score": rating_score,
        "text_score": text_score,
        "lang_score": lang_score,
    }


def recommend_for_profile(reviews_df, profile):
    rows = []
    for _, row in reviews_df.iterrows():
        if len(str(row["text_orig"]).strip()) < 12:
            continue
        score, details = score_review(row, profile)
        reason_parts = []
        if details["local_score"] >= 0.7:
            reason_parts.append("local/exotic fit")
        if details["oil_score"] >= 0.7:
            reason_parts.append("oil fit")
        if details["spice_score"] >= 0.7:
            reason_parts.append("spice fit")
        if details["herb_score"] >= 0.7:
            reason_parts.append("herb fit")
        if details["text_score"] >= 0.25:
            reason_parts.append("keyword overlap")
        rows.append({
            "gmap_id": row["gmap_id"],
            "text": row["text_orig"],
            "lang_group": row["lang_group"],
            "rating": row["rating"],
            "score": score,
            "details": details,
            "reason": ", ".join(reason_parts) if reason_parts else "general fit",
        })
    out = pd.DataFrame(rows).sort_values(["score", "rating"], ascending=[False, False]).head(10).reset_index(drop=True)
    return out


def make_profile(name, profile):
    return {
        "name": name,
        "local_pref": profile[0],
        "oil_pref": profile[1],
        "spice_pref": profile[2],
        "herb_pref": profile[3],
        "prefer_ko_reviews": True,
    }


def write_markdown_report(profile_results, reviews_df):
    out_path = OUT_DIR / "personalized_demo_report.md"
    lines = []
    lines.append("# Personalized Review Recommendation Demo")
    lines.append("")
    lines.append("## Why this is a valid ML-style service")
    lines.append("- We transform user survey answers into a preference vector and rank reviews by taste-profile similarity.")
    lines.append("- Reviews are first tagged by language and taste cues, then filtered and ranked so the extension can present the most relevant evidence to the user.")
    lines.append("- This is similar to retrieval + ranking systems used in recommendation products, where user preferences and item attributes are matched.")
    lines.append("")
    lines.append("## Data foundations")
    lines.append(f"- Number of reviews used in demo: {len(reviews_df):,}")
    lines.append(f"- Language distribution: {dict(reviews_df['lang_group'].value_counts().to_dict())}")
    lines.append("")
    lines.append("## Sample user profiles")
    for profile_name, profile in profile_results.items():
        lines.append(f"### {profile_name}")
        lines.append(f"- Survey vector: local={profile['profile']['local_pref']}, oil={profile['profile']['oil_pref']}, spice={profile['profile']['spice_pref']}, herb={profile['profile']['herb_pref']}")
        lines.append("- Top 3 recommended reviews:")
        for row in profile["top_reviews"][:3]:
            lines.append(f"  - score={row['score']:.3f}, lang={row['lang_group']}, rating={row['rating']}, reason={row['reason']}, text={row['text'][:180]}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote markdown report to {out_path}")


def make_profile_plot(profile_results):
    # A simple comparison of the top review score by profile
    labels = list(profile_results.keys())
    scores = [profile_results[name]["top_reviews"][0]["score"] for name in labels]
    plt.figure(figsize=(8, 4.5))
    sns.barplot(x=labels, y=scores, palette="viridis")
    plt.title("Top review score by sample profile")
    plt.ylabel("Top review score")
    plt.xlabel("Profile")
    plt.tight_layout()
    out_path = PLOTS_DIR / "personalized_demo_profile_scores.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote plot to {out_path}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reviews_df = load_data()

    profiles = [
        make_profile("현지식 선호형", [5, 3, 4, 5]),
        make_profile("무난한 맛 선호형", [2, 2, 2, 2]),
        make_profile("매운/향 강한 맛 선호형", [4, 4, 5, 5]),
    ]

    results = {}
    for profile in profiles:
        recs = recommend_for_profile(reviews_df, profile)
        results[profile["name"]] = {
            "profile": profile,
            "top_reviews": recs[["text", "lang_group", "rating", "score", "reason"]].to_dict(orient="records"),
        }

    write_markdown_report(results, reviews_df)
    make_profile_plot(results)

    # also save a JSON view for downstream use
    out_json = OUT_DIR / "personalized_demo_results.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote JSON demo output to {out_json}")


if __name__ == "__main__":
    main()
