"""
01_filter_la_places_v2_1.py

v2 문제: 카테고리 중 하나라도 exclude면 탈락시켜서
  Bottega Louie / Olive Garden / Chick-fil-A / Canter's / Perch 등
  'Caterer','Event venue','Banquet hall','Chocolate shop' 부가태그가 붙은
  명백한 식당 1,706곳(리뷰 50+)이 잘려나갔음.

v2.1 해법: category 배열의 첫 원소 = 주 업종(primary).
  판정 계층:
    1) primary가 include  -> 통과
    2) primary가 exclude  -> 탈락
    3) primary가 neutral  -> 나머지에 include 있고 exclude 없을 때만 통과
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
    "shopping mall", "department store", "discount store", "discount supermarket",
    "natural goods store", "seafood market", "japanese grocery store",
    "korean grocery store", "asian grocery store", "amusement center",
    "delivery service", "tourist attraction",
}
EXCLUDE_SUBSTRING = [
    "manufacturer", "supplier", "wholesaler", "exporter", "distributor",
    "store fixture", "equipment", "church", "school", "clinic",
    "grocery store", "supermarket",
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
    "eatery", "bar", "pub", "tavern", "brewpub", "cantina", "crab", "lobster",
    "taqueria", "pizzeria", "trattoria", "osteria", "creperie", "patisserie",
    "dumpling", "hot pot", "shabu", "poke", "wings", "waffle", "pancake",
    "soup", "salad", "vegan", "vegetarian", "halal", "kosher", "pho", "curry",
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
    "birrieria", "pupuseria", "panaderia", "marisqueria", "cevicheria",
    "juice bar", "crab house", "oyster bar", "sushi bar", "boba shop",
}


def normalize(cat: str) -> str:
    return re.sub(r"\s+", " ", cat.strip().lower())


def classify_category(cat_norm: str) -> str:
    if cat_norm in INCLUDE_EXACT:
        return "include"
    if cat_norm in EXCLUDE_EXACT:
        return "exclude"
    for kw in EXCLUDE_SUBSTRING:
        if kw in cat_norm:
            return "exclude"
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
    primary = labels[0]
    if primary == "include":
        return True, "primary_include"
    if primary == "exclude":
        return False, "primary_exclude"
    rest = labels[1:]
    if "include" in rest and "exclude" not in rest:
        return True, "secondary_include"
    if "include" in rest and "exclude" in rest:
        return False, "secondary_conflict"
    return False, "neutral_only"


def is_in_la(lat, lon):
    if lat is None or lon is None:
        return False
    return (LA_LAT_RANGE[0] <= lat <= LA_LAT_RANGE[1]) and \
           (LA_LON_RANGE[0] <= lon <= LA_LON_RANGE[1])


def misc_signals(rec):
    misc = rec.get("MISC") or {}
    so = " ".join(misc.get("Service options", []) or []).lower()
    return {
        "_primary_category": (rec.get("category") or [None])[0],
        "_has_dinein": "dine-in" in so,
        "_has_takeout": "takeout" in so,
        "_has_delivery": "delivery" in so,
        "_has_price": bool(rec.get("price")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-reviews", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n_total = n_la = n_pass_cat = n_final = n_dup = 0
    seen = set()
    reason_counter = Counter(); primary_pass = Counter(); primary_fail = Counter()

    out_meta = out_dir / "la_restaurants_meta_v21.jsonl"
    out_ids = out_dir / "la_restaurant_gmap_ids_v21.txt"

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
            prim = normalize(cats[0]) if cats else "(none)"
            if not ok:
                primary_fail[prim] += 1
                continue
            n_pass_cat += 1
            primary_pass[prim] += 1
            if (rec.get("num_of_reviews") or 0) < args.min_reviews:
                reason_counter["below_min_reviews"] += 1
                continue
            gid = rec.get("gmap_id")
            if gid in seen:
                n_dup += 1
                continue
            seen.add(gid)
            rec.update(misc_signals(rec))
            n_final += 1
            fmeta.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fids.write(gid + "\n")
            if n_total % 200_000 == 0:
                print(f"  ... {n_total:,}곳 스캔 (통과 {n_final:,})", flush=True)

    audit_path = out_dir / "primary_category_audit.csv"
    with open(audit_path, "w", encoding="utf-8-sig") as f:
        f.write("primary_category,decision,n_passed,n_failed\n")
        allc = set(primary_pass) | set(primary_fail)
        rows = [(c, classify_category(c), primary_pass.get(c, 0), primary_fail.get(c, 0)) for c in allc]
        rows.sort(key=lambda r: -(r[2] + r[3]))
        for c, d, p, q in rows:
            f.write(f'"{c}",{d},{p},{q}\n')

    print("\n=== 필터링 결과 (v2.1: primary category 기준) ===")
    print(f"  전체 스캔 업체:        {n_total:,}")
    print(f"  LA bbox 내:            {n_la:,}")
    print(f"  카테고리 통과:         {n_pass_cat:,}")
    print(f"  최소리뷰({args.min_reviews})까지 통과: {n_final:,}")
    print(f"  중복 gmap_id 제거:     {n_dup:,}")
    print("\n  [판정 사유 분포]")
    for k, v in reason_counter.most_common():
        print(f"    {k}: {v:,}")
    print(f"\n메타: {out_meta}")
    print(f"ID  : {out_ids}")
    print(f"감사: {audit_path}")


if __name__ == "__main__":
    main()
