# External Imports
from langchain_core.messages import HumanMessage, AIMessage
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

    # 1. Initialize chat history in Streamlit session state if empty
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    rag_index = load_index()
    set_index(rag_index)

    model_name = st.selectbox("Select a model:", MODELS, index=0)
    agent = build_agent(model_name)

    # 2. Render all past conversation bubbles so they stay on screen on refresh
    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(text)

    # 3. Modernize input using st.chat_input
    if question := st.chat_input('Ask a course question...'):
        # Render user prompt immediately
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append(("user", question))

    # 4. Convert simple session history to LangChain message instances
    graph_input_messages = []
    for role, text in st.session_state.chat_history:
        if role == "user":
            graph_input_messages.append(HumanMessage(content=text))
        else:
            graph_input_messages.append(AIMessage(content=text))

    # 5. Process through the graph loop
    result = None
    with st.chat_message("assistant"):
        try:
          result = agent.invoke(
              {'messages': graph_input_messages},
              config={"recursion_limit": 20},
          )

          # Target the final conversational string meant for the user
          final_ai_msg = next(
              (m for m in reversed(result['messages'])
               if isinstance(m, AIMessage) and m.content),
              None
          )

          final_response = final_ai_msg.content if final_ai_msg else "No response."
          st.markdown(final_response)
        except Exception as e:
            st.error(f"An execution thread error occurred: {e}")
            final_response = \
              "Sorry, I encountered an internal error processing your request."

    # Save answer to state history array and flush details to log files
    st.session_state.chat_history.append(("assistant", final_response))
    if result is not None:
        _write_result_to_log(result)


if __name__ == "__main__":
    run()
