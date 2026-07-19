import streamlit as st
from llm_interface import answer_with_llm
from course_data import build_index
from course_rag import retrieve
from dotenv import load_dotenv

load_dotenv()

def ask(question, index):
  retrieved = retrieve(question, index)
  for item in retrieved:
    print(item['score'], item['course'].get('Code'))
  ans = answer_with_llm(question, retrieved)
  return ans

st.set_page_config(page_title="UCSC Course Assistant")
st.title("UCSC Course Assistant")

index = build_index()

question = st.text_input("Enter your question:")
if question:
  answer = ask(question, index)
  st.subheader("Answer:")
  st.write(answer)