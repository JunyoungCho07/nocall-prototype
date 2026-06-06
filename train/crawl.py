"""
제품 이미지 웹크롤링 스크립트.

사용법:
    uv run python train/crawl.py

동작:
  - DuckDuckGo 이미지 검색 (icrawler 사용, 로그인 불필요)
  - 각 제품별 100장씩 train/data/images/train/<class>/ 에 저장
  - 이미 다운로드된 파일은 건너뜀

이후 수동 작업:
  - LabelImg로 bounding box 라벨링 (YOLO format 저장)
  - 라벨 파일을 train/data/labels/train/ 에 배치
"""

from pathlib import Path

try:
    from icrawler.builtin import GoogleImageCrawler
except ImportError:
    import subprocess, sys
    subprocess.run(["uv", "add", "icrawler"], check=True)
    from icrawler.builtin import GoogleImageCrawler

QUERIES = {
    "cola":       ["코카콜라 캔", "coca cola can", "콜라 캔"],
    "water":      ["생수 페트병", "물병 투명", "water bottle plastic"],
    "cereal-box": ["시리얼 박스", "cereal box", "켈로그 박스"],
    "paper-cup":  ["종이컵", "paper cup disposable", "dixie cup"],
    "cup-noodle": ["컵라면", "cup noodle instant", "컵누들"],
}

COUNT_PER_QUERY = 35  # 쿼리 3개 × 35장 ≈ 100장/제품
BASE_DIR = Path(__file__).parent / "data" / "images" / "train"


def crawl():
    for cls_name, queries in QUERIES.items():
        out_dir = BASE_DIR / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(out_dir.glob("*.jpg"))) + len(list(out_dir.glob("*.png")))
        print(f"\n[{cls_name}] 기존 {existing}장 확인")

        for query in queries:
            crawler = GoogleImageCrawler(storage={"root_dir": str(out_dir)})
            crawler.crawl(keyword=query, max_num=COUNT_PER_QUERY, file_idx_offset="auto")

        total = len(list(out_dir.glob("*.jpg"))) + len(list(out_dir.glob("*.png")))
        print(f"[{cls_name}] 총 {total}장 수집 완료 → {out_dir}")

    print("\n=== 크롤링 완료 ===")
    print("다음 단계: LabelImg로 bounding box 라벨링")
    print("  uv tool install labelimg")
    print("  labelimg train/data/images/train train/data/labels/train")


if __name__ == "__main__":
    crawl()
