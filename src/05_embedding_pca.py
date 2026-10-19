"""
05_embedding_pca.py

리뷰 텍스트를 벡터화한 뒤 PCA를 적용해:
  1. scree plot -> n_components 결정 근거
  2. 한/영 리뷰 분리 여부 2D 시각화

--method tfidf        : 언어별 별도 TF-IDF (오프라인 동작)
--method multilingual : LaBSE 등 다국어 임베딩으로 동일 공간 비교 (인터넷 필요)

사용법:
  python 05_embedding_pca.py \
      --reviews-path ../data/processed/la_restaurant_reviews_lang.jsonl \
      --method multilingual \
      --out-dir ../outputs
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def load_reviews(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("lang_final") in ("ko", "en"):
                rows.append(rec)
    return pd.DataFrame(rows)


def vectorize_tfidf(df):
    from sklearn.feature_extraction.text import TfidfVectorizer

    results = {}
    for lang in ["ko", "en"]:
        sub = df[df["lang_final"] == lang]
        if len(sub) < 5:
            continue
        vec = TfidfVectorizer(max_features=500, min_df=2)
        X = vec.fit_transform(sub["text"]).toarray()
        results[lang] = (sub, X)
    return results


def run_pca_and_plot(X, labels, title_prefix, out_dir: Path):
    n_components = min(10, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(6, 4))
    explained = pca.explained_variance_ratio_
    cum = np.cumsum(explained)
    ax.bar(range(1, len(explained) + 1), explained, alpha=0.6, label="Explained variance (individual)")
    ax.plot(range(1, len(explained) + 1), cum, color="red", marker="o", label="Cumulative explained variance")
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=1, label="80% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title(f"{title_prefix} - Scree Plot")
    ax.legend()
    fig.tight_layout()
    scree_path = out_dir / f"{title_prefix}_scree.png"
    fig.savefig(scree_path, dpi=120)
    plt.close(fig)

    n_for_80pct = int(np.searchsorted(cum, 0.8) + 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    for lab in sorted(set(labels)):
        mask = np.array(labels) == lab
        ax.scatter(coords[mask, 0], coords[mask, 1], alpha=0.5, s=15, label=lab)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% explained)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% explained)" if len(explained) > 1 else "PC2")
    ax.set_title(f"{title_prefix} - PCA 2D")
    ax.legend()
    fig.tight_layout()
    scatter_path = out_dir / f"{title_prefix}_pca2d.png"
    fig.savefig(scatter_path, dpi=120)
    plt.close(fig)

    return {
        "n_components_for_80pct_variance": n_for_80pct,
        "explained_variance_ratio_top5": explained[:5].round(3).tolist(),
        "scree_plot": str(scree_path),
        "scatter_plot": str(scatter_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews-path", type=str, required=True)
    ap.add_argument("--method", type=str, choices=["tfidf", "multilingual"], default="tfidf")
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_reviews(args.reviews_path)
    print(f"로드된 리뷰: {len(df)}건 (ko={sum(df.lang_final=='ko')}, en={sum(df.lang_final=='en')})")

    if args.method == "tfidf":
        by_lang = vectorize_tfidf(df)
        for lang, (sub, X) in by_lang.items():
            labels = (sub["rating"] >= 4).map({True: "rating>=4", False: "rating<4"}).tolist()
            stats = run_pca_and_plot(X, labels, f"tfidf_{lang}", out_dir)
            print(f"\n[{lang}] PCA 결과:")
            for k, v in stats.items():
                print(f"  {k}: {v}")

    elif args.method == "multilingual":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("pip install sentence-transformers 먼저 실행하세요.")
            return
        print("LaBSE 모델 로드 중 (최초 실행 시 다운로드)...")
        model = SentenceTransformer("sentence-transformers/LaBSE")
        X = model.encode(df["text"].tolist(), show_progress_bar=True)
        labels = df["lang_final"].tolist()
        stats = run_pca_and_plot(np.array(X), labels, "multilingual_ko_en", out_dir)
        print("\n[다국어 임베딩] PCA 결과 (한/영 동일 공간 비교):")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
