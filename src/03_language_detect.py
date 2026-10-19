"""
03_language_detect.py

각 리뷰의 작성 언어를 판별한다.
- langdetect: 통계 기반 언어 감지
- 한글 문자 비율 규칙: 짧은 텍스트 오분류 보정 (한글 비율 규칙 우선)

사용법:
  python 03_language_detect.py \
      --in-path ../data/processed/la_restaurant_reviews.jsonl \
      --out-path ../data/processed/la_restaurant_reviews_lang.jsonl
"""
import argparse
import json
import re
from pathlib import Path

from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 42

HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
ALPHA_RE = re.compile(r"[A-Za-z\uac00-\ud7a3]")

HANGUL_RATIO_THRESHOLD = 0.3


def hangul_ratio(text: str) -> float:
    alpha_chars = ALPHA_RE.findall(text)
    if not alpha_chars:
        return 0.0
    hangul_chars = HANGUL_RE.findall(text)
    return len(hangul_chars) / len(alpha_chars)


def detect_language(text: str) -> dict:
    ratio = hangul_ratio(text)

    try:
        ld_result = detect(text)
    except LangDetectException:
        ld_result = None

    if ratio >= HANGUL_RATIO_THRESHOLD:
        final = "ko"
        method = "hangul_ratio"
    elif ld_result == "ko":
        final = "ko"
        method = "langdetect"
    elif ld_result == "en":
        final = "en"
        method = "langdetect"
    else:
        final = "other"
        method = "langdetect_other" if ld_result else "undetected"

    return {
        "lang_final": final,
        "lang_method": method,
        "hangul_ratio": round(ratio, 3),
        "langdetect_raw": ld_result,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-path", type=str, required=True)
    ap.add_argument("--out-path", type=str, required=True)
    args = ap.parse_args()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"ko": 0, "en": 0, "other": 0}
    disagreements = 0

    with open(args.in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            rec = json.loads(line)
            lang_info = detect_language(rec["text"])
            rec.update(lang_info)

            if lang_info["langdetect_raw"] is not None and \
               lang_info["langdetect_raw"] != lang_info["lang_final"] and \
               lang_info["lang_final"] in ("ko", "en"):
                disagreements += 1

            counts[lang_info["lang_final"]] = counts.get(lang_info["lang_final"], 0) + 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if i % 200_000 == 0:
                print(f"  ... {i:,}건 처리")

    print("=== 언어 분리 결과 ===")
    for k, v in counts.items():
        print(f"  {k}: {v:,}")
    print(f"  langdetect vs 한글비율 규칙 불일치: {disagreements:,}")
    print(f"저장 위치: {out_path}")


if __name__ == "__main__":
    main()
