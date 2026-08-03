# External Imports
from langchain_core.messages import HumanMessage
import logging
import streamlit as st
# Internal Imports
from ucsc_course_assistant.agent.graph import build_agent
from ucsc_course_assistant.agent.logger_config import setup_logging
from ucsc_course_assistant.course_data import load_index
from ucsc_course_assistant.agent.tools import set_index

# initialize logging once at startup
setup_logging()
logger = logging.getLogger(__name__)
logger.info('Running agent...')

MODELS = (
    'nvidia/nemotron-3-ultra-550b-a55b:free',
    # 'openai/gpt-oss-120b:free',
    # 'meta-llama/llama-3.3-70b-instruct:free',
    # 'openai/gpt-oss-20b:free',
    # 'meta-llama/llama-3.2-3b-instruct:free',
)


def _write_result_to_log(result):
    for msg in result['messages']:
        logger.info(f"\n--- {msg.type} ---")
        logger.info(f"content: {msg.content}")

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            logger.info(f"tool_calls: {msg.tool_calls}")

        if msg.type == "tool":
            logger.info(f"tool_call_id: {msg.tool_call_id}")
            logger.info(f"tool_name: {msg.name}")


def run():
    st.set_page_config(page_title="UCSC Course Assistant")
    st.title("UCSC Course Assistant")

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
        _write_result_to_log(result)


if __name__ == "__main__":
    run()
