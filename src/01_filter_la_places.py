"""
01_filter_la_places.py

meta-California.json.gz 에서
- LA 지역 위경도 범위
- 음식점(restaurant) 카테고리
- 최소 리뷰 수(--min-reviews)
조건을 만족하는 업체만 필터링해 gmap_id 목록 + 메타데이터를 저장한다.

메모리 전체 로드를 피하기 위해 한 줄씩 스트리밍으로 읽는다.

사용법:
  python 01_filter_la_places.py \
      --meta-path ../data/raw/meta-California.json.gz \
      --out-path ../data/processed/la_restaurants_meta.jsonl \
      --min-reviews 50
"""
import argparse
import gzip
import json
from pathlib import Path

LA_LAT_RANGE = (33.7, 34.35)
LA_LON_RANGE = (-118.7, -118.1)

RESTAURANT_KEYWORDS = [
    "restaurant", "korean bbq", "barbecue", "cafe", "diner",
    "bistro", "eatery", "food", "noodle", "sushi", "bar & grill",
]


def is_restaurant(categories):
    if not categories:
        return False
    joined = " ".join(categories).lower()
    return any(kw in joined for kw in RESTAURANT_KEYWORDS)


def is_in_la(lat, lon):
    if lat is None or lon is None:
        return False
    return (LA_LAT_RANGE[0] <= lat <= LA_LAT_RANGE[1]) and \
           (LA_LON_RANGE[0] <= lon <= LA_LON_RANGE[1])


def stream_filter(meta_path: Path, min_reviews: int):
    n_total = 0
    n_la = 0
    n_restaurant = 0
    n_final = 0
    results = []

    with gzip.open(meta_path, "rt", encoding="utf-8") as f:
        for line in f:
            n_total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            lat, lon = rec.get("latitude"), rec.get("longitude")
            if not is_in_la(lat, lon):
                continue
            n_la += 1

            if not is_restaurant(rec.get("category")):
                continue
            n_restaurant += 1

            if (rec.get("num_of_reviews") or 0) < min_reviews:
                continue
            n_final += 1

            results.append(rec)

            if n_total % 2_000_000 == 0:
                print(f"  ... {n_total:,}곳 스캔")

    stats = {
        "total_places_scanned": n_total,
        "in_la_bbox": n_la,
        "restaurant_category": n_restaurant,
        "passed_min_reviews": n_final,
    }
    return results, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-path", type=str, required=True)
    ap.add_argument("--out-path", type=str, required=True)
    ap.add_argument("--min-reviews", type=int, default=50)
    args = ap.parse_args()

    meta_path = Path(args.meta_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results, stats = stream_filter(meta_path, args.min_reviews)

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("=== 필터링 결과 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"저장 위치: {out_path}")

    gmap_ids_path = out_path.with_name("la_restaurant_gmap_ids.txt")
    with open(gmap_ids_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(rec["gmap_id"] + "\n")
    print(f"gmap_id 목록: {gmap_ids_path}")


if __name__ == "__main__":
    main()
