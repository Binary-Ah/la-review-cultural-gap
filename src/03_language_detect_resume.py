"""
03_language_detect_resume.py

03_language_detect.py의 개선판.

지난 실행에서 확인된 문제:
  - 서버 스왑이 100% 소진될 정도의 메모리 압박 상황에서 OOM Killer로
    프로세스가 강제 종료(SIGKILL)됨
  - 로그가 stdout 버퍼링 때문에 즉시 flush 안 돼서, 죽은 시점에
    어디까지 처리됐는지 로그로 확인 불가능했음

개선 사항:
  1. --resume 플래그: 이미 만들어진 출력 파일의 줄 수를 세서,
     입력 파일에서 그만큼 건너뛰고 이어서 처리 (처음부터 재시작 안 해도 됨)
  2. 모든 print에 flush=True + 로그를 훨씬 자주(10,000건마다) 남겨서
     죽어도 어디까지 됐는지 바로 알 수 있음
  3. 출력 파일도 주기적으로 명시적 flush()해서, 프로세스가 죽어도
     디스크에 이미 쓰인 데이터는 안전하게 남도록 함

사용법 (처음 실행):
  python 03_language_detect_resume.py \
      --in-path ../data/processed/la_restaurant_reviews.jsonl \
      --out-path ../data/processed/la_restaurant_reviews_lang.jsonl

사용법 (중단 후 이어서):
  python 03_language_detect_resume.py \
      --in-path ../data/processed/la_restaurant_reviews.jsonl \
      --out-path ../data/processed/la_restaurant_reviews_lang.jsonl \
      --resume
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
LOG_EVERY = 10_000
FLUSH_EVERY = 10_000


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


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-path", type=str, required=True)
    ap.add_argument("--out-path", type=str, required=True)
    ap.add_argument("--resume", action="store_true",
                     help="출력 파일이 이미 있으면 그 줄 수만큼 입력에서 건너뛰고 이어서 처리")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    already_done = 0
    file_mode = "w"
    if args.resume and out_path.exists():
        already_done = count_lines(out_path)
        file_mode = "a"
        print(f"[재개 모드] 이미 처리된 {already_done:,}줄을 건너뛰고 이어서 처리합니다.", flush=True)
    elif args.resume:
        print("[재개 모드] 출력 파일이 없어 처음부터 시작합니다.", flush=True)

    counts = {"ko": 0, "en": 0, "other": 0}
    disagreements = 0
    n_processed_this_run = 0

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, file_mode, encoding="utf-8") as fout:

        # 이미 처리된 줄 건너뛰기
        for _ in range(already_done):
            next(fin, None)

        for i, line in enumerate(fin, start=already_done + 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            lang_info = detect_language(rec.get("text", ""))
            rec.update(lang_info)

            if lang_info["langdetect_raw"] is not None and \
               lang_info["langdetect_raw"] != lang_info["lang_final"] and \
               lang_info["lang_final"] in ("ko", "en"):
                disagreements += 1

            counts[lang_info["lang_final"]] = counts.get(lang_info["lang_final"], 0) + 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_processed_this_run += 1

            if i % LOG_EVERY == 0:
                print(f"  ... 누적 {i:,}줄 처리 (이번 실행에서 {n_processed_this_run:,}줄)", flush=True)

            if i % FLUSH_EVERY == 0:
                fout.flush()

        fout.flush()

    print("=== 언어 분리 결과 (이번 실행분 기준) ===", flush=True)
    for k, v in counts.items():
        print(f"  {k}: {v:,}", flush=True)
    print(f"  langdetect vs 한글비율 규칙 불일치: {disagreements:,}", flush=True)
    print(f"저장 위치: {out_path}", flush=True)
    print(f"총 누적 처리 줄 수: {already_done + n_processed_this_run:,}", flush=True)


if __name__ == "__main__":
    main()
