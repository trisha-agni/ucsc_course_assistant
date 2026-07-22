import json
import requests
import streamlit as st
import torch
from bs4 import BeautifulSoup, Tag
from concurrent.futures import ThreadPoolExecutor, as_completed
from course_rag import embed_text
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE_PATH = ROOT / "data" / "course_data.jsonl"
INDEX_FILE_PATH = ROOT / "data" / "course_index.pt"
FETCH_ALL_COURSES = True
BASE_URL = "https://catalog.ucsc.edu/en/current/general-catalog/courses"
COURSE_URL_SUBSTR = "/en/current/general-catalog/courses/"
MAX_WORKERS = 32

TEST_URLs = [
    "https://catalog.ucsc.edu/en/current/general-catalog/courses/cse-computer-science-and-engineering/lower-division/cse-30",  # noqa: E501
    "https://catalog.ucsc.edu/en/current/general-catalog/courses/cmpm-computational-media/upper-division/cmpm-146",  # noqa: E501
    "https://catalog.ucsc.edu/en/current/general-catalog/courses/cse-computer-science-and-engineering/upper-division/cse-101",  # noqa: E501
    "https://catalog.ucsc.edu/en/current/general-catalog/courses/stat-statistics/upper-division/stat-131",  # noqa: E501
    "https://catalog.ucsc.edu/en/current/general-catalog/courses/cse-computer-science-and-engineering/lower-division/cse-40",  # noqa: E501
    "https://catalog.ucsc.edu/en/current/general-catalog/courses/cse-computer-science-and-engineering/lower-division/cse-12",  # noqa: E501
]

URL_KEY = "URL"
CODE_KEY = "Code"
TITLE_KEY = "Title"
DESC_KEY = "Description"
REQ_KEY = "Requirements"
GEN_ED_KEY = "General Education Code"
CREDITS_KEY = "Credits"


def clean(txt):
  text = ' '.join(txt.split())
  replacements = {
      " ,": ",",
      " .": ".",
      " ;": ";",
      " :": ":",
      " )": ")",
      " (": "(",
  }
  for old, new in replacements.items():
    text = text.replace(old, new)
  return text


def _create_soup(url):
  resp = requests.get(url, timeout=20)
  resp.raise_for_status()
  soup = BeautifulSoup(resp.text, 'html.parser')
  soup = soup.select_one('main') or soup.body or soup
  return soup


def _parse_heading(soup, course_data):
  h1 = soup.find('h1')
  heading = clean(h1.get_text(' ', strip=True)) if h1 else None
  code, title = '', heading
  if heading:
    parts = heading.split(maxsplit=2)
    if len(parts) == 3:
      code, title = ' '.join(parts[:2]), parts[2]
  course_data[CODE_KEY] = code
  course_data[TITLE_KEY] = title


def _parse_desc(soup, course_data):
  desc = soup.select_one(".desc")
  if not desc:
    course_data[DESC_KEY] = ""
    return
  course_data[DESC_KEY] = clean(desc.get_text(" ", strip=True))


def _parse_extra_fields(soup, course_data):
  all_fields = soup.select('div.extraFields') + soup.select('div.genEd')
  for field in all_fields:
    heading = field.find(['h2', 'h3', 'h4', 'h5', 'h6'])
    if not heading:
      continue
    key = clean(heading.get_text(' ', strip=True))
    values = []
    for child in field.children:
      if not isinstance(child, Tag):
        text = clean(str(child))
        if text:
          values.append(text)
        continue
      if child is heading:
        continue
      text = clean(child.get_text(' ', strip=True))
      if text:
        values.append(text)
    course_data[key] = ' '.join(values)


def parse_course_url(url):
  soup = _create_soup(url)
  course_data = {URL_KEY: url}
  _parse_heading(soup, course_data)
  _parse_desc(soup, course_data)
  _parse_extra_fields(soup, course_data)
  return course_data


@st.cache_resource
def load_index():
  return torch.load(INDEX_FILE_PATH, weights_only=False)


def course_to_rag_text(d):
  return f"""
  Course: {d.get(CODE_KEY, '')} - {d.get(TITLE_KEY, '')}
  Credits: {d.get(CREDITS_KEY, '')}
  General Education Code: {d.get(GEN_ED_KEY, '')}

  Description: {d.get(DESC_KEY, '')}

  Requirements: {d.get(REQ_KEY, '')}

  Source:
  {d.get(URL_KEY, '')}
  """


def _discover_urls(base_url):
  html = requests.get(base_url, timeout=20).text
  soup = BeautifulSoup(html, "html.parser")
  urls = set()
  for a in soup.select("a[href]"):
    href = a["href"]
    full_url = urljoin(base_url, href)
    if COURSE_URL_SUBSTR in full_url:
      urls.add(full_url)
  return sorted(urls)


def _get_course_urls():
  if not FETCH_ALL_COURSES:
    return TEST_URLs
  dept_urls = _discover_urls(BASE_URL)
  print(f"discovered {len(dept_urls)} course urls")
  course_urls = set()
  """for dept_url in dept_urls:
    urls = _discover_urls(dept_url)
    for url in urls:
      course_urls.add(url)"""
  with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exexcutor:
    future_to_dept = {exexcutor.submit(_discover_urls, url): url for url in dept_urls}
    for future in as_completed(future_to_dept):
      dept_url = future_to_dept[future]
      num_trials = 5
      while num_trials > 0:
        try:
          urls = future.result()
          course_urls.update(urls)
          break
        except Exception as e:
          num_trials -= 1
          if num_trials == 0:
            print(f"skipped {dept_url} due to error: {e}")
  print(f"discovered {len(course_urls)} course urls")
  return sorted(course_urls)


def _save_index(loaded_course_data):
  all_course_rag_text = [course_to_rag_text(d) for d in loaded_course_data]
  # Stack the list of tensors into a single tensor
  all_course_embs = torch.stack([embed_text(t) for t in all_course_rag_text])
  index = {
    'courses': loaded_course_data,
    'texts': all_course_rag_text,
    'embeddings': all_course_embs,
  }
  torch.save(index, INDEX_FILE_PATH)


def save_course_data():
  DATA_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
  print(f"Saving course data to {DATA_FILE_PATH}")
  nc = 0
  course_urls = _get_course_urls()
  parsed_results = []
  """with DATA_FILE_PATH.open("w", encoding="utf-8") as f:
      for url in course_urls:
        d = parse_course_url(url)
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
        nc += 1"""
  with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_url = {executor.submit(parse_course_url, url): url for url in course_urls}
    for future in as_completed(future_to_url):
      url = future_to_url[future]
      num_trials = 5
      while num_trials > 0:
        try:
          d = future.result()
          parsed_results.append(d)
          nc += 1
          if nc % 500 == 0:
            print(f"Parsed {nc} courses so far...")
          break
        except Exception as e:
          num_trials -= 1
          if num_trials == 0:
            print(f"skipped {url} due to error: {e}")
    parsed_results.sort(key=lambda x: x.get(URL_KEY, ""))
    with DATA_FILE_PATH.open("w", encoding="utf-8") as f:
      for d in parsed_results:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
    _save_index(parsed_results)
  print(f"Saved course data for {len(parsed_results)} courses.")


if __name__ == "__main__":
  save_course_data()
