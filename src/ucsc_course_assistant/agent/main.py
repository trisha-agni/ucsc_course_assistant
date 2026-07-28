# External Imports
import streamlit as st
# Internal Imports
from ucsc_course_assistant.course_data import load_index
from ucsc_course_assistant.agent.tools import set_index

MODELS = (
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-coder:free",
)


rag_index = load_index()
set_index(rag_index)

model_name = st.selectbox("Select a model:", MODELS, index=0)
agent = build_agent(model_name)


if __name__ == "___main__":
    pass