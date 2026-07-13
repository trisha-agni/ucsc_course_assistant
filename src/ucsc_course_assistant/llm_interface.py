import ollama
import os
import streamlit as st
from openai import OpenAI


CHAT_MODEL_NAME = 'llama3.1:8b'
OPENROUTER_MODEL_NAME = 'openai/gpt-oss-20b:free'
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'

#@st.cache_resource
def get_free_model_ids():
  print("calling get free model ids")
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

USE_OPENROUTER = True
orclient = None
if USE_OPENROUTER and not OPENROUTER_API_KEY:
  raise RuntimeError('OPENROUTER_API_KEY is not set')
elif orclient is not None:
  orclient = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
  )

def answer_with_ollama(question, prompt):
  response = ollama.chat(model=CHAT_MODEL_NAME,
                         messages=[{'role': 'user', 'content': prompt}],
                         options={'temperature': 0})
  return response['message']['content']

def answer_with_openrouter(question, prompt):
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
          max_tokens=500,
      )
      resp = response.choices[0].message.content
      if resp is not None:
        return resp
    except Exception as e:
      last_err = e
      print('Error: model: ', model_id)
  raise last_err

def answer_with_llm(question, retrieved):
  prompt = _create_prompt(question, retrieved)
  if USE_OPENROUTER:
    return answer_with_openrouter(question, prompt)
  else:
    return answer_with_ollama(question, prompt)