"""
02_extract_reviews.py

review-California.json.gz (전체 리뷰, 7천만 건 규모)에서
01단계 gmap_id 목록에 해당하는 리뷰만 스트리밍으로 추출한다.

사용법:
  python 02_extract_reviews.py \
      --review-path ../data/raw/review-California.json.gz \
      --gmap-ids-path ../data/processed/la_restaurant_gmap_ids.txt \
      --out-path ../data/processed/la_restaurant_reviews.jsonl
"""
import argparse
import gzip
import json
from pathlib import Path


def load_gmap_ids(path: Path) -> set:
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def stream_extract(review_path: Path, gmap_ids: set, out_path: Path):
    n_total = 0
    n_matched = 0
    n_empty_text = 0

    with gzip.open(review_path, "rt", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            n_total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if rec.get("gmap_id") not in gmap_ids:
                continue

            text = rec.get("text")
            if not text or not text.strip():
                n_empty_text += 1
                continue

            n_matched += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if n_total % 5_000_000 == 0:
                print(f"  ... {n_total:,}건 스캔, {n_matched:,}건 매칭")

    return {"total_reviews_scanned": n_total,
            "matched": n_matched,
            "skipped_empty_text": n_empty_text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-path", type=str, required=True)
    ap.add_argument("--gmap-ids-path", type=str, required=True)
    ap.add_argument("--out-path", type=str, required=True)
    args = ap.parse_args()

    gmap_ids = load_gmap_ids(Path(args.gmap_ids_path))
    print(f"대상 gmap_id 수: {len(gmap_ids)}")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = stream_extract(Path(args.review_path), gmap_ids, out_path)

    print("=== 추출 결과 ===")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")
    print(f"저장 위치: {out_path}")


if __name__ == "__main__":
    main()
