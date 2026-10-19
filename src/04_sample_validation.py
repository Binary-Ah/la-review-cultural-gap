"""
04_sample_validation.py

장소(gmap_id)별 한국어/영어 리뷰 수를 집계하고,
--min-ko-reviews 기준을 만족하는 장소만 최종 분석 대상으로 확정한다.

사용법:
  python 04_sample_validation.py \
      --reviews-path ../data/processed/la_restaurant_reviews_lang.jsonl \
      --meta-path ../data/processed/la_restaurants_meta.jsonl \
      --min-ko-reviews 50 \
      --out-dir ../data/processed
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews-path", type=str, required=True)
    ap.add_argument("--meta-path", type=str, required=True)
    ap.add_argument("--min-ko-reviews", type=int, default=50)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name_map = {}
    with open(args.meta_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            name_map[rec["gmap_id"]] = rec.get("name", "")

    counts = defaultdict(lambda: {"ko": 0, "en": 0, "other": 0})
    with open(args.reviews_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            counts[rec["gmap_id"]][rec["lang_final"]] += 1

    rows = []
    for gmap_id, c in counts.items():
        rows.append({
            "gmap_id": gmap_id,
            "name": name_map.get(gmap_id, ""),
            "ko_reviews": c["ko"],
            "en_reviews": c["en"],
            "other_reviews": c["other"],
            "total_reviews": c["ko"] + c["en"] + c["other"],
            "passes_threshold": c["ko"] >= args.min_ko_reviews,
        })

    df = pd.DataFrame(rows).sort_values("ko_reviews", ascending=False)

    all_path = out_dir / "place_language_counts_all.csv"
    df.to_csv(all_path, index=False, encoding="utf-8-sig")

    final_df = df[df["passes_threshold"]].copy()
    final_path = out_dir / "place_language_counts_FINAL.csv"
    final_df.to_csv(final_path, index=False, encoding="utf-8-sig")

    final_gmap_ids_path = out_dir / "final_gmap_ids.txt"
    with open(final_gmap_ids_path, "w", encoding="utf-8") as f:
        for gid in final_df["gmap_id"]:
            f.write(gid + "\n")

    print("=== 표본 사전 검증 결과 ===")
    print(f"  전체 후보 장소 수: {len(df)}")
    print(f"  기준(한국어 리뷰 >= {args.min_ko_reviews}건) 통과 장소 수: {len(final_df)}")
    print(f"  제외된 장소 수: {len(df) - len(final_df)}")
    print()
    print(df.to_string(index=False))
    print()
    print(f"전체 집계: {all_path}")
    print(f"최종 확정 장소: {final_path}")
    print(f"최종 gmap_id 목록: {final_gmap_ids_path}")

    if len(final_df) < 5:
        print("\n[경고] 기준을 통과한 장소가 5곳 미만입니다.")
        print("       --min-ko-reviews 값을 낮추거나 필터를 재검토하세요.")


if __name__ == "__main__":
    main()
