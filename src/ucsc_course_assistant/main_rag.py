import streamlit as st
from llm_interface import answer_with_llm
from course_data import format_course_sources, load_index
from course_rag import retrieve
from dotenv import load_dotenv

load_dotenv()


def ask(question, index):
  retrieved = retrieve(question, index)
  course_scores = [
    (r['course'], r['score'])
    for r in retrieved
  ]
  ans = answer_with_llm(question, retrieved)
  return ans, format_course_sources(course_scores)


st.set_page_config(page_title="UCSC Course Assistant")
st.title("UCSC Course Assistant")

index = load_index()

question = st.text_input("Enter your question:")
if question:
  answer, format_sources = ask(question, index)
  st.subheader("Answer:")
  st.write(answer)
  st.subheader("Sources:")
  st.write(format_sources)
