"""
06_rebuild_final_dataset.py

새 필터(v2)로 바뀐 업체 집합 기준으로 최종 분석 파일을 조립한다.
핵심: 이미 오래 돌린 03단계(언어분리) 결과를 버리지 않는 것.

  v1 ∩ v2 : 이미 언어분리 끝남 -> 재사용
  v2 - v1 : 신규 업체 -> 02+03 추가 실행분을 합침
  v1 - v2 : 비음식점 등 -> 버림
"""
import argparse
import json
from pathlib import Path


def load_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return set(l.strip() for l in f if l.strip())


def cmd_diff(args):
    old = load_ids(args.old_ids)
    new = load_ids(args.new_ids)
    added, kept, dropped = new - old, new & old, old - new

    print("=== gmap_id 집합 비교 ===")
    print(f"  기존(v1) 업체 수 : {len(old):,}")
    print(f"  신규(v2) 업체 수 : {len(new):,}")
    print("  --------------------------------")
    print(f"  유지(교집합)     : {len(kept):,}  <- 재사용")
    print(f"  신규 추가        : {len(added):,}  <- 02+03 추가 실행 필요")
    print(f"  제외됨(v1에만)   : {len(dropped):,}  <- 버림")

    out = Path(args.out_added)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for gid in sorted(added):
            f.write(gid + "\n")
    print(f"\n추가 추출 대상 id: {out}")
    if new:
        print(f"기존 작업 재사용률: {len(kept)/len(new)*100:.1f}%")


def cmd_build(args):
    new_ids = load_ids(args.new_ids)
    n_read = n_written = n_skipped = n_dup = 0
    lang_counter = {}
    seen_keys = set()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for lf in args.lang_files:
            p = Path(lf)
            if not p.exists():
                print(f"  [건너뜀] 파일 없음: {lf}")
                continue
            n_file = 0
            with open(p, "r", encoding="utf-8") as fin:
                for line in fin:
                    n_read += 1
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("gmap_id") not in new_ids:
                        n_skipped += 1
                        continue
                    key = (rec.get("user_id"), rec.get("gmap_id"), rec.get("time"))
                    if key in seen_keys:
                        n_dup += 1
                        continue
                    seen_keys.add(key)
                    lang = rec.get("lang_final", "unknown")
                    lang_counter[lang] = lang_counter.get(lang, 0) + 1
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_written += 1
                    n_file += 1
            print(f"  {p.name}: {n_file:,}건 채택")

    print("\n=== 최종 데이터셋 조립 결과 ===")
    print(f"  읽은 리뷰 총합      : {n_read:,}")
    print(f"  v2 대상 아님(제외)  : {n_skipped:,}")
    print(f"  중복 제거           : {n_dup:,}")
    print(f"  최종 리뷰 수        : {n_written:,}")
    print("\n  [언어 분포]")
    for k, v in sorted(lang_counter.items(), key=lambda x: -x[1]):
        pct = v / n_written * 100 if n_written else 0
        print(f"    {k}: {v:,} ({pct:.2f}%)")
    print(f"\n저장: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diff")
    d.add_argument("--old-ids", required=True)
    d.add_argument("--new-ids", required=True)
    d.add_argument("--out-added", required=True)
    d.set_defaults(func=cmd_diff)
    b = sub.add_parser("build")
    b.add_argument("--new-ids", required=True)
    b.add_argument("--lang-files", nargs="+", required=True)
    b.add_argument("--out-path", required=True)
    b.set_defaults(func=cmd_build)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
