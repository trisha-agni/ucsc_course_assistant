import json
import requests
import streamlit as st
import torch
from bs4 import BeautifulSoup, Tag
from course_rag import embed_text
from pathlib import Path

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
def get_course_data():
  with DATA_FILE_PATH.open("r", encoding="utf-8") as f:
    all_course_data = [json.loads(l) for l in f if l.strip()]
  return all_course_data

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

@st.cache_resource
def build_index():
  loaded_course_data = get_course_data()
  all_course_rag_text = [course_to_rag_text(d) for d in loaded_course_data]
  # Stack the list of tensors into a single tensor
  all_course_embs = torch.stack([embed_text(t) for t in all_course_rag_text])
  return {
    'courses': loaded_course_data,
    'texts': all_course_rag_text,
    'embeddings': all_course_embs,
  }

def save_course_data():
  DATA_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
  print(f"Saving course data to {DATA_FILE_PATH}")
  nc = 0
  with DATA_FILE_PATH.open("w", encoding="utf-8") as f:
    for url in URLs:
      d = parse_course_url(url)
      f.write(json.dumps(d, ensure_ascii=False) + "\n")
      nc += 1
  print(f"Saved course data for {nc} courses.")

if __name__ == "__main__":
  save_course_data()