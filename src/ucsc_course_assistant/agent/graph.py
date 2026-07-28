# External Imports
from langchain_core.messages import SystemMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
# Internal Imports
from ucsc_course_assistant.agent.prompts import SYSTEM_PROMPT
from ucsc_course_assistant.agent.tools import get_tools


def _create_llm(model_name):
    return ChatOpenRouter(
        model=model_name,
        temperature=0,
    ).bind_tools(get_tools())


def build_agent(model_name):
    llm = _create_llm(model_name)

    def _call_llm(state: MessagesState):
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
        return {'messages': [llm.invoke(msgs)]}

    graph = StateGraph(MessagesState)
    graph.add_node('llm', _call_llm)
    graph.add_node('tools', ToolNode(get_tools()))
    graph.add_edge(START, 'llm')
    graph.add_conditional_edges('llm', tools_condition)
    graph.add_edge('tools', 'llm')

    return graph.compile()
