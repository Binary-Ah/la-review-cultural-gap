import pathlib
import re
import json
import math
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression

sns.set(style='whitegrid')

# Lexicons for simple attribute scoring
OILY_WORDS = [
    'greasy','oily','fatty','grease','greasey','greasy','buttery','rich','heavy','cream','fried','deep-fried','油','fat','grease','油っぽ',
]
SPICY_WORDS = [
    'spicy','hot','chili','chilli','pepper','spice','辛','매운','spicy','heat','jalapeno','habanero','sichuan','태','고추','칠리'
]
HERB_WORDS = [
    'cilantro','coriander','herb','herby','basil','rosemary','thyme','parsley','oregano','mint','허브','고수','cumin','카다멈'
]
EXOTIC_WORDS = [
    'exotic','unique','authentic','local','traditional','unusual','weird','foreign','이국','전통','현지','authentic'
]

WORD_GROUPS = {
    'oily': OILY_WORDS,
    'spicy': SPICY_WORDS,
    'herb': HERB_WORDS,
    'exotic': EXOTIC_WORDS,
}

RE_WORD = re.compile(r"[\w\u4e00-\u9fff\u00C0-\u017F']+", flags=re.UNICODE)


def lexicon_score(text, lexicon):
    if not isinstance(text, str) or not text:
        return 0.0
    words = RE_WORD.findall(text.lower())
    if not words:
        return 0.0
    c = 0
    for token in lexicon:
        c += sum(1 for w in words if token in w)
    return c / len(words)


def compute_review_attributes(reviews_path, sample_frac=None, out_parquet=None):
    print('Loading reviews...')
    df = pd.read_parquet(reviews_path, columns=['gmap_id','text_orig','lang_group','rating'])
    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=42)
    print('Computing lexicon-based attribute scores...')
    for name, lex in WORD_GROUPS.items():
        df[name + '_score'] = df['text_orig'].fillna('').apply(lambda t: lexicon_score(t, lex))
    if out_parquet:
        df.to_parquet(out_parquet, index=False)
    return df


def aggregate_by_category(df_reviews, cat_csv):
    cat = pd.read_csv(cat_csv)
    # cat has primary_category and counts; need map from gmap_id to primary_category
    # We can load meta mapping from processed meta file if available
    try:
        meta = pd.read_json('data/processed/la_restaurants_meta_v21.jsonl', lines=True)
        meta_map = meta.set_index('gmap_id')['_primary_category'].to_dict()
        df_reviews['primary_category'] = df_reviews['gmap_id'].map(meta_map)
    except Exception:
        # fallback: leave primary_category NaN
        df_reviews['primary_category'] = None
    agg = df_reviews.groupby('primary_category').agg(
        reviews_n=('text_orig','count'),
        oily_mean=('oily_score','mean'),
        spicy_mean=('spicy_score','mean'),
        herb_mean=('herb_score','mean'),
        exotic_mean=('exotic_score','mean'),
        ko_n=('lang_group', lambda s: (s=='ko').sum()),
        en_n=('lang_group', lambda s: (s=='en').sum()),
        rating_mean=('rating','mean'),
    ).reset_index()
    return agg


def plot_top_attributes(agg, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for attr in ['oily_mean','spicy_mean','herb_mean','exotic_mean']:
        top = agg.sort_values(attr, ascending=False).head(10)
        plt.figure(figsize=(8,5))
        sns.barplot(x=attr, y='primary_category', data=top, palette='viridis')
        plt.title(f'Top 10 categories by {attr}')
        plt.xlabel(attr)
        plt.ylabel('Primary category')
        fn = out_dir/f'top10_{attr}.png'
        plt.tight_layout()
        plt.savefig(fn, dpi=150)
        plt.close()
        print('Wrote', fn)


def plot_relgap_vs_attr(cat_csv, agg, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cat = pd.read_csv(cat_csv)
    merged = cat.merge(agg, how='left', left_on='primary_category', right_on='primary_category')
    for attr in ['spicy_mean','oily_mean','herb_mean']:
        df = merged.dropna(subset=[attr,'rel_gap'])
        if df.empty:
            continue
        plt.figure(figsize=(6,5))
        sns.regplot(x=attr, y='rel_gap', data=df, scatter_kws={'s':10}, line_kws={'color':'red'})
        plt.title(f'rel_gap vs {attr}')
        plt.xlabel(attr)
        plt.ylabel('rel_gap (ko - en adjusted)')
        fn = out_dir/f'relgap_vs_{attr}.png'
        plt.tight_layout()
        plt.savefig(fn, dpi=150)
        plt.close()
        print('Wrote', fn)
        # print correlation
        corr = df[attr].corr(df['rel_gap'])
        print(f'Correlation rel_gap vs {attr}:', corr)


def plot_tsne_reviews(df_reviews, out_dir, n_samples=5000):
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = df_reviews.dropna(subset=['text_orig']).sample(n=min(n_samples, len(df_reviews)), random_state=42)
    texts = sample['text_orig'].fillna('').tolist()
    vect = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1,2), max_features=20000)
    X = vect.fit_transform(texts)
    svd = TruncatedSVD(n_components=50, random_state=42)
    Xred = svd.fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42, perplexity=40)
    X2 = tsne.fit_transform(Xred)
    sample['tsne_x'] = X2[:,0]
    sample['tsne_y'] = X2[:,1]
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='tsne_x', y='tsne_y', hue='lang_group', data=sample, s=8, alpha=0.8, palette='Set1', legend='full')
    plt.title('t-SNE of reviews TF-IDF (sample) colored by lang_group')
    fn = out_dir/'tsne_lang_group.png'
    plt.tight_layout()
    plt.savefig(fn, dpi=150)
    plt.close()
    print('Wrote', fn)


def train_quick_lang_model(df_reviews, out_dir):
    # small balanced sample
    frames = []
    for lab in ['ko','en','other']:
        s = df_reviews[df_reviews['lang_group']==lab]
        frames.append(s.sample(n=min(20000, len(s)), random_state=42))
    df = pd.concat([f for f in frames if len(f)>0])
    X = df['text_orig'].fillna('')
    y = df['lang_group']
    vect = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1,2), max_features=20000, strip_accents='unicode')
    X_tf = vect.fit_transform(X)
    clf = LogisticRegression(solver='saga', max_iter=800, class_weight='balanced')
    clf.fit(X_tf, y)
    # get top tokens per class
    feature_names = np.array(vect.get_feature_names_out())
    for idx, lab in enumerate(clf.classes_):
        top_idx = np.argsort(clf.coef_[idx])[-20:][::-1]
        toks = feature_names[top_idx]
        weights = clf.coef_[idx][top_idx]
        plt.figure(figsize=(8,5))
        sns.barplot(x=weights, y=toks, color='C0')
        plt.title(f'Top tokens for class {lab}')
        plt.xlabel('Coefficient')
        fn = out_dir/f'lang_top_tokens_{lab}.png'
        plt.tight_layout()
        plt.savefig(fn, dpi=150)
        plt.close()
        print('Wrote', fn)


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    reviews_path = root / 'data' / 'processed' / 'reviews_final_th03.parquet'
    cat_csv = root / 'data' / 'processed' / 'category_stats_th03.csv'
    out_dir = root / 'outputs' / 'plots'

    df_reviews = compute_review_attributes(str(reviews_path), sample_frac=0.1)
    agg = aggregate_by_category(df_reviews, str(cat_csv))
    plot_top_attributes(agg, out_dir)
    plot_relgap_vs_attr(str(cat_csv), agg, out_dir)
    plot_tsne_reviews(df_reviews, out_dir, n_samples=3000)
    train_quick_lang_model(df_reviews, out_dir)

    # write a short demo markdown
    demo_md = root / 'outputs' / 'ml_demo_explanations.md'
    text = []
    text.append('# Demo: 개인화용 리뷰 검색/요약 — 분석 결과')
    text.append('\n')
    text.append('생성된 플롯은 `outputs/plots/`에 있습니다. 주요 인사이트:')
    text.append('- 특정 업종에서 `spice`/`oily`/`herb` 스코어가 상대적으로 높게 관찰됩니다. (top10 시각화)')
    text.append('- `rel_gap`(ko-en 조정 격차)와 `spicy_mean`/`oily_mean` 사이의 상관관계를 확인해, 특정 맛 프로파일이 한국인 평점에 미치는 영향을 검토할 수 있습니다.')
    text.append('- 언어별 t-SNE 시각화는 `ko/en/other` 리뷰가 의미 공간에서 어느 정도 분리되는지를 보여주어, 텍스트 기반 ML이 경계 사례를 포착할 타당성을 시각적으로 제공합니다.')
    demo_md.write_text('\n'.join(text), encoding='utf-8')
    print('Wrote demo explanations to', demo_md)

if __name__=='__main__':
    main()
