"""
01_filter_la_places_v2.py

v1 문제: category substring 매칭("food","cafe")으로
  Textile exporter / Computer store / Church / Meat Market / Winery /
  Snack food manufacturer 등이 대량 오통과했고,
  반대로 Taqueria / Pizzeria / Bakery / Steak house / Donut shop 등
  실제 음식점은 누락됐음.

v2 개선:
  1. 정규화 후 '정확 매칭 + 접미사 규칙'으로 판정 (substring 금지)
  2. 명시적 제외 사전을 먼저 적용
  3. gmap_id 중복 제거
  4. MISC/price는 보조 신호로 기록만 (판정에 미사용, 사후 검수용)
  5. category_audit.csv 출력 -> 카테고리별 통과/탈락 전량 기록
"""
import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

LA_LAT_RANGE = (33.7, 34.35)
LA_LON_RANGE = (-118.7, -118.1)

EXCLUDE_EXACT = {
    "textile exporter", "fabric store", "computer store", "computer service",
    "pentecostal church", "church", "food bank", "food manufacturer",
    "food producer", "food products supplier", "snack food manufacturer",
    "meat market", "butcher shop", "wine store", "liquor store", "winery",
    "brewery", "distillery", "farmers' market", "farmers market",
    "grocery store", "supermarket", "convenience store", "gas station",
    "warehouse store", "wholesaler", "food wholesaler",
    "fruit and vegetable store", "produce market", "health food store",
    "vitamin & supplements store", "candy store", "chocolate shop",
    "catering food and drink supplier", "caterer", "hotel", "motel",
    "banquet hall", "event venue", "vending machine supplier",
    "restaurant supply store", "kitchen supply store",
}
EXCLUDE_SUBSTRING = [
    "manufacturer", "supplier", "wholesaler", "exporter", "distributor",
    "store fixture", "equipment", "church", "school", "clinic",
]

INCLUDE_SUFFIX = (
    " restaurant", " restaurants", " house", " shop", " stand", " truck",
    " bar", " grill", " bbq", " buffet", " joint", " parlor", " parlour",
)
FOOD_HINTS = {
    "restaurant", "food", "cafe", "coffee", "tea", "pizza", "burger", "taco",
    "sushi", "ramen", "noodle", "bbq", "barbecue", "grill", "steak", "seafood",
    "chicken", "sandwich", "deli", "bakery", "dessert", "ice cream", "yogurt",
    "donut", "doughnut", "juice", "smoothie", "boba", "bubble", "kitchen",
    "diner", "bistro", "buffet", "brunch", "breakfast", "lunch", "dinner",
    "eatery", "bar", "pub", "tavern", "brewpub", "cantina",
    "taqueria", "pizzeria", "trattoria", "osteria", "creperie", "patisserie",
    "dumpling", "hot pot", "shabu", "poke", "wings", "waffle", "pancake",
    "soup", "salad", "vegan", "vegetarian", "halal", "kosher",
}
INCLUDE_EXACT = {
    "taqueria", "pizzeria", "trattoria", "osteria", "creperie", "brasserie",
    "cafe", "cafeteria", "coffee shop", "coffee store", "espresso bar",
    "tea house", "bubble tea store", "juice shop", "smoothie shop",
    "bakery", "patisserie", "pastry shop", "cake shop", "donut shop",
    "doughnut shop", "bagel shop", "sandwich shop", "deli", "delicatessen",
    "diner", "bistro", "gastropub", "pub", "bar", "bar & grill", "wine bar",
    "cocktail bar", "sports bar", "brewpub", "beer garden", "tavern",
    "ice cream shop", "frozen yogurt shop", "gelato shop", "dessert shop",
    "dessert restaurant", "snack bar", "food court", "food stand",
    "food truck", "hot dog stand", "hamburger restaurant", "steak house",
    "sushi takeaway", "ramen shop", "noodle shop", "poke bar", "buffet",
    "meal takeaway", "meal delivery", "takeout restaurant",
}


def normalize(cat: str) -> str:
    return re.sub(r"\s+", " ", cat.strip().lower())


def classify_category(cat_norm: str) -> str:
    if cat_norm in EXCLUDE_EXACT:
        return "exclude"
    for kw in EXCLUDE_SUBSTRING:
        if kw in cat_norm:
            return "exclude"
    if cat_norm in INCLUDE_EXACT:
        return "include"
    if cat_norm.endswith(" restaurant") or cat_norm == "restaurant":
        return "include"
    for suf in INCLUDE_SUFFIX:
        if cat_norm.endswith(suf):
            if any(h in cat_norm for h in FOOD_HINTS):
                return "include"
    return "neutral"


def is_restaurant(categories):
    if not categories:
        return False, "no_category"
    labels = [classify_category(normalize(c)) for c in categories]
    if "exclude" in labels:
        return False, "excluded_category"
    if "include" in labels:
        return True, "included_category"
    return False, "neutral_only"


def is_in_la(lat, lon):
    if lat is None or lon is None:
        return False
    return (LA_LAT_RANGE[0] <= lat <= LA_LAT_RANGE[1]) and \
           (LA_LON_RANGE[0] <= lon <= LA_LON_RANGE[1])


def misc_signals(rec):
    misc = rec.get("MISC") or {}
    service_opts = " ".join(misc.get("Service options", []) or []).lower()
    return {
        "_has_dinein": "dine-in" in service_opts,
        "_has_takeout": "takeout" in service_opts,
        "_has_delivery": "delivery" in service_opts,
        "_has_price": bool(rec.get("price")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-path", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--min-reviews", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = n_la = n_pass_cat = n_final = n_dup = 0
    seen_gmap_ids = set()
    reason_counter = Counter()
    cat_pass = Counter()
    cat_fail = Counter()

    out_meta = out_dir / "la_restaurants_meta_v2.jsonl"
    out_ids = out_dir / "la_restaurant_gmap_ids_v2.txt"

    with gzip.open(args.meta_path, "rt", encoding="utf-8") as fin, \
         open(out_meta, "w", encoding="utf-8") as fmeta, \
         open(out_ids, "w", encoding="utf-8") as fids:

        for line in fin:
            n_total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not is_in_la(rec.get("latitude"), rec.get("longitude")):
                continue
            n_la += 1

            cats = rec.get("category") or []
            ok, reason = is_restaurant(cats)
            reason_counter[reason] += 1

            if not ok:
                for c in cats:
                    cat_fail[normalize(c)] += 1
                continue
            n_pass_cat += 1
            for c in cats:
                cat_pass[normalize(c)] += 1

            if (rec.get("num_of_reviews") or 0) < args.min_reviews:
                reason_counter["below_min_reviews"] += 1
                continue

            gid = rec.get("gmap_id")
            if gid in seen_gmap_ids:
                n_dup += 1
                continue
            seen_gmap_ids.add(gid)

            rec.update(misc_signals(rec))
            n_final += 1
            fmeta.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fids.write(gid + "\n")

            if n_total % 200_000 == 0:
                print(f"  ... {n_total:,}곳 스캔 (통과 {n_final:,})", flush=True)

    audit_path = out_dir / "category_audit.csv"
    with open(audit_path, "w", encoding="utf-8-sig") as f:
        f.write("category,decision,n_passed_businesses,n_failed_businesses\n")
        all_cats = set(cat_pass) | set(cat_fail)
        rows = [(c, classify_category(c), cat_pass.get(c, 0), cat_fail.get(c, 0))
                for c in all_cats]
        rows.sort(key=lambda r: -(r[2] + r[3]))
        for c, d, p, q in rows:
            f.write(f'"{c}",{d},{p},{q}\n')

    print("\n=== 필터링 결과 (v2) ===")
    print(f"  전체 스캔 업체:        {n_total:,}")
    print(f"  LA bbox 내:            {n_la:,}")
    print(f"  카테고리 통과:         {n_pass_cat:,}")
    print(f"  최소리뷰({args.min_reviews})까지 통과: {n_final:,}")
    print(f"  중복 gmap_id 제거:     {n_dup:,}")
    print("\n  [탈락 사유 분포]")
    for k, v in reason_counter.most_common():
        print(f"    {k}: {v:,}")
    print(f"\n메타: {out_meta}")
    print(f"ID  : {out_ids}")
    print(f"감사: {audit_path}")


if __name__ == "__main__":
    main()
