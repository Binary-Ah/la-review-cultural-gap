"""
check_progress.py

파이프라인 각 단계의 산출물 파일이 존재하는지, 몇 줄인지 확인해서
"우리 지금 어디까지 했지?"를 한 번에 보여주는 스크립트.

사용법:
  cd review-project
  python src/check_progress.py
"""
import gzip
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

def count_lines(path: Path, gz=False):
    if not path.exists():
        return None
    try:
        if gz or path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
    except Exception as e:
        return f"에러: {e}"


def check_process_running(script_name: str) -> bool:
    try:
        out = subprocess.run(["pgrep", "-f", script_name], capture_output=True, text=True)
        return bool(out.stdout.strip())
    except Exception:
        return False


STAGES = [
    {
        "name": "0. 원본 데이터",
        "files": [
            ("meta-California.json.gz", BASE / "data/raw/meta-California.json.gz"),
            ("review-California.json.gz", BASE / "data/raw/review-California.json.gz"),
        ],
        "process": None,
    },
    {
        "name": "1. LA 음식점 메타 필터링",
        "files": [
            ("la_restaurants_meta.jsonl", BASE / "data/processed/la_restaurants_meta.jsonl"),
            ("la_restaurant_gmap_ids.txt", BASE / "data/processed/la_restaurant_gmap_ids.txt"),
        ],
        "process": "01_filter_la_places.py",
    },
    {
        "name": "2. 리뷰 추출",
        "files": [
            ("la_restaurant_reviews.jsonl", BASE / "data/processed/la_restaurant_reviews.jsonl"),
        ],
        "process": "02_extract_reviews.py",
    },
    {
        "name": "3. 언어 분리",
        "files": [
            ("la_restaurant_reviews_lang.jsonl", BASE / "data/processed/la_restaurant_reviews_lang.jsonl"),
        ],
        "process": "03_language_detect.py",
    },
    {
        "name": "4. 표본 사전 검증 (최종 장소 확정)",
        "files": [
            ("place_language_counts_FINAL.csv", BASE / "data/processed/place_language_counts_FINAL.csv"),
            ("final_gmap_ids.txt", BASE / "data/processed/final_gmap_ids.txt"),
        ],
        "process": "04_sample_validation.py",
    },
    {
        "name": "5. PCA / 임베딩 시각화",
        "files": [
            ("outputs 폴더 내 png", BASE / "outputs"),
        ],
        "process": "05_embedding_pca.py",
    },
]


def main():
    print("=" * 60)
    print("파이프라인 진행 상황 체크")
    print("=" * 60)

    for stage in STAGES:
        proc_running = check_process_running(stage["process"]) if stage["process"] else False
        print(f"\n[{stage['name']}]")
        if proc_running:
            print("  상태: 🔄 지금 백그라운드에서 실행 중")

        any_exists = False
        for label, path in stage["files"]:
            if path.is_dir():
                pngs = list(path.glob("*.png"))
                if pngs:
                    any_exists = True
                    print(f"  {label}: ✅ {len(pngs)}개 파일 존재 ({', '.join(p.name for p in pngs[:3])}{'...' if len(pngs) > 3 else ''})")
                else:
                    print(f"  {label}: ❌ 없음")
                continue

            if path.exists():
                any_exists = True
                size_mb = path.stat().st_size / (1024 * 1024)
                n_lines = count_lines(path)
                line_info = f", {n_lines:,}줄" if isinstance(n_lines, int) else ""
                print(f"  {label}: ✅ 존재 ({size_mb:.1f}MB{line_info})")
            else:
                print(f"  {label}: ❌ 없음")

        if not any_exists and not proc_running:
            print("  -> 아직 이 단계 시작 안 함 (또는 실행이 죽었을 수 있음)")

    print("\n" + "=" * 60)
    print("완료. 이 출력 전체를 복사해서 팀/Claude에게 공유하면 상태 파악 빠름.")


if __name__ == "__main__":
    main()
