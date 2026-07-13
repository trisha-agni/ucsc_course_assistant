import json
from pathlib import Path
from course_data import get_course_data, parse_course_url

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE_PATH = ROOT / "data" / "course_data.jsonl"
URLs = [
    "https://catalog.ucsc.edu/en/current/general-catalog/courses/cse-computer-science-and-engineering/lower-division/cse-30",
    "https://catalog.ucsc.edu/en/current/general-catalog/courses/cmpm-computational-media/upper-division/cmpm-146",
    "https://catalog.ucsc.edu/en/current/general-catalog/courses/cse-computer-science-and-engineering/upper-division/cse-101",
    "https://catalog.ucsc.edu/en/current/general-catalog/courses/stat-statistics/upper-division/stat-131",
    "https://catalog.ucsc.edu/en/current/general-catalog/courses/cse-computer-science-and-engineering/lower-division/cse-40",
    "https://catalog.ucsc.edu/en/current/general-catalog/courses/cse-computer-science-and-engineering/lower-division/cse-12",
]

def save_courses():
    with DATA_FILE_PATH.open("w", encoding="utf-8") as f:
        for url in URLs:
            d = parse_course_url(url)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    save_courses()