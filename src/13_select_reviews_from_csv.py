import pandas as pd
import numpy as np
import pathlib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / 'data' / 'processed' / 'the_cheezecake_factory.csv'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

LEX = {
    'oily':['greasy','oil','rich','butter','buttery','heavy','fried','fatty','느끼','기름'],
    'spicy':['spicy','hot','chili','매운','고추','칠리'],
    'herb':['cilantro','herb','basil','고수','허브','향신료'],
    'exotic':['authentic','local','unique','이국','현지','전통']
}
WORD_RE = re.compile(r"[\w\u4e00-\u9fff']+", re.UNICODE)

def score_lex(text, lex):
    if not isinstance(text,str):
        return 0.0
    toks = WORD_RE.findall(text.lower())
    if not toks:
        return 0.0
    cnt=0
    for k in lex:
        cnt += sum(1 for t in toks if k in t)
    return cnt/len(toks)


def main():
    # try attachment location if not in data/processed
    p1 = ROOT / 'data' / 'processed' / 'the cheezecake factory.csv'
    p2 = ROOT / 'data' / 'processed' / 'the_cheezecake_factory.csv'
    p3 = ROOT / 'the cheezecake factory.csv'
    candidates = [p1,p2,p3]
    csv=None
    for p in candidates:
        if p.exists():
            csv=p
            break
    if csv is None:
        # try fallback to Downloads path from attachment
        dl = pathlib.Path.home() / 'Downloads' / 'the cheezecake factory.csv'
        if dl.exists():
            csv=dl
    if csv is None:
        print('CSV not found. Place the file at', p2)
        return
    print('Using',csv)
    df = pd.read_csv(csv)
    df['text'] = df['text'].astype(str)
    # compute lex scores
    for name,lex in LEX.items():
        df[name+'_score'] = df['text'].apply(lambda t: score_lex(t,lex))
    # simple tag for language: has hangul?
    df['has_hangul'] = df['text'].str.contains(r'[가-힣]')
    # TF-IDF
    vect = TfidfVectorizer(min_df=1, stop_words='english')
    X = vect.fit_transform(df['text'])
    # sample profiles as pseudo-queries
    profiles = {
        'Local-pref': 'authentic local traditional',
        'Mild-pref': 'mild fresh light',
        'Rich-pref': 'rich buttery heavy creamy',
        'Spicy-pref': 'spicy hot chili pepper',
    }
    qvecs = vect.transform(list(profiles.values()))
    sims = cosine_similarity(qvecs, X)
    # collect top examples per profile
    selected = {}
    for i,(k,q) in enumerate(profiles.items()):
        idxs = np.argsort(-sims[i])[:5]
        selected[k] = df.iloc[idxs][['author','date','text','has_hangul','oily_score','spicy_score','herb_score','exotic_score']].copy()
        selected[k]['sim'] = sims[i,idxs]
    # also pick extremes per lexicon
    extremes = {}
    for col in ['oily_score','spicy_score','herb_score','exotic_score']:
        top = df.sort_values(col, ascending=False).head(5)
        extremes[col] = top[['author','date','text',col]].copy()
    # write outputs
    for k,tab in selected.items():
        fn = OUT / f'selected_{k.replace(" ","_")}.csv'
        tab.to_csv(fn, index=False)
        print('Wrote',fn)
    for k,tab in extremes.items():
        fn = OUT / f'extreme_{k}.csv'
        tab.to_csv(fn, index=False)
        print('Wrote',fn)
    # plot similarity heatmap between profiles and top 10 reviews
    top10_idxs = np.argsort(-sims.mean(axis=0))[:10]
    sim_sub = cosine_similarity(qvecs, X[top10_idxs])
    plt.figure(figsize=(6,3))
    sns.heatmap(sim_sub, annot=True, xticklabels=[df['text'].iloc[i][:30] for i in top10_idxs], yticklabels=list(profiles.keys()))
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(OUT / 'profile_review_similarity_heatmap.png', dpi=150)
    print('Wrote similarity heatmap')

if __name__=='__main__':
    main()
