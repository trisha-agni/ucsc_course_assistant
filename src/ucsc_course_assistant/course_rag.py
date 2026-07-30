import streamlit as st
import torch
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
DEFAULT_NUM_RESULTS = 32


@st.cache_resource
def load_embedding_model():
  return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_text(text):
  embed_model = load_embedding_model()
  return embed_model.encode(text,
                            normalize_embeddings=True,
                            convert_to_tensor=True)


def retrieve(query, index, num_results=DEFAULT_NUM_RESULTS):
  q_embed = embed_text(query)
  scores = torch.matmul(index['embeddings'], q_embed)  # cosine similarity
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
