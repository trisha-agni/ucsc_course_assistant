# External Imports
from langchain_core.messages import HumanMessage
import streamlit as st
# Internal Imports
from ucsc_course_assistant.agent.graph import build_agent
from ucsc_course_assistant.course_data import load_index
from ucsc_course_assistant.agent.tools import set_index

MODELS = (
    'nvidia/nemotron-3-ultra-550b-a55b:free',
    # 'openai/gpt-oss-120b:free',
    # 'meta-llama/llama-3.3-70b-instruct:free',
    # 'openai/gpt-oss-20b:free',
    # 'meta-llama/llama-3.2-3b-instruct:free',
)


rag_index = load_index()
set_index(rag_index)

model_name = st.selectbox("Select a model:", MODELS, index=0)
agent = build_agent(model_name)
question = st.text_input("Ask a course question")
if question:
    result = agent.invoke({
        'messages': [HumanMessage(content=question)],
    })
    st.write(result['messages'][-1].content)


if __name__ == "___main__":
    pass