# section 1
from bs4 import BeautifulSoup, Tag
import requests

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

def get_course_data():
  all_course_data = []
  for url in URLs:
    d = parse_course_url(url)
    all_course_data.append(d)
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

def get_free_model_ids():
  import requests

  URL = "https://openrouter.ai/api/v1/models"

  resp = requests.get(URL, timeout=20)
  resp.raise_for_status()

  FREE_MODEL_IDS = []
  models = resp.json()['data']
  for model in models:
    id = model['id']
    if id.endswith(':free'):
      FREE_MODEL_IDS.append(id)
  return FREE_MODEL_IDS

# section 2
import ollama
import os
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

DEFAULT_NUM_RESULTS = 100
CHAT_MODEL_NAME = 'llama3.1:8b'
#CHAT_MODEL_NAME = 'phi3:mini'
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

load_dotenv()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
OPENROUTER_MODEL_NAME = 'openai/gpt-oss-20b:free'
USE_OPENROUTER = True

if USE_OPENROUTER and not OPENROUTER_API_KEY:
  raise RuntimeError('OPENROUTER_API_KEY is not set')

embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
orclient = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

def embed_text(text):
  return embed_model.encode(text,
                            normalize_embeddings=True,
                            convert_to_tensor=True)

def build_index(loaded_course_data):
  all_course_rag_text = [course_to_rag_text(d) for d in loaded_course_data]
  # Stack the list of tensors into a single tensor
  all_course_embs = torch.stack([embed_text(t) for t in all_course_rag_text])
  return {
      'courses': loaded_course_data,
      'texts': all_course_rag_text,
      'embeddings': all_course_embs,
  }

def retrieve(query, index, num_results=DEFAULT_NUM_RESULTS):
  q_embed = embed_text(query)
  scores = torch.matmul(index['embeddings'], q_embed) # cosine similarity
  top_results = torch.topk(scores, k=min(num_results, len(index['courses'])))
  results = []
  for score, idx in zip(top_results.values, top_results.indices):
    idx = int(idx.item())
    results.append({
        'score': score.item(),
        'course': index['courses'][idx],
        'text': index['texts'][idx],
    })
  return results

def _create_prompt(question, retrieved):
  context = '\n\n-----\n\n'.join(item['text'] for item in retrieved)
  prompt = f"""
  You are UC Santa Cruz course assistant.

  Answer the question using ONLY the course information below.
  If the answer is not present in the course information, say:
  "I don't know from the provided course records."

  Course information:
  {context}

  Question:
  {question}
  """.strip()
  return prompt

def answer_with_ollama(question, retrieved):
  prompt = _create_prompt(question, retrieved)
  response = ollama.chat(model=CHAT_MODEL_NAME,
                         messages=[{'role': 'user', 'content': prompt}],
                         options={'temperature': 0})
  return response['message']['content']

def answer_with_openrouter(question, retrieved):
  prompt = _create_prompt(question, retrieved)
  last_err = None
  FREE_MODEL_IDS = get_free_model_ids()
  FREE_MODEL_IDS = [OPENROUTER_MODEL_NAME] + FREE_MODEL_IDS
  for model_id in FREE_MODEL_IDS:
    try:
      response = orclient.chat.completions.create(
          #model=OPENROUTER_MODEL_NAME,
          model=model_id,
          messages=[{'role': 'user', 'content': prompt}],
          temperature=0,
          max_tokens=250,
      )
      resp = response.choices[0].message.content
      if resp is not None:
        return resp
    except Exception as e:
      last_err = e
      print('Error: model: ', model_id)
  raise last_err

def ask(question, index):
  retrieved = retrieve(question, index)
  for item in retrieved:
    print(item['score'], item['course'].get('Code'))
  if USE_OPENROUTER:
    ans = answer_with_openrouter(question, retrieved)
  else:
    ans = answer_with_ollama(question, retrieved)
  return ans

loaded_course_data = get_course_data()
index = build_index(loaded_course_data)
q = input("Enter your question: ")
print(ask(q, index))