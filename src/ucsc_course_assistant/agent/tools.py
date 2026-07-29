# External Imports
from langchain_core.tools import tool
import json
# Internal Imports
from ucsc_course_assistant.course_data import (
    CODE_KEY, TITLE_KEY, DESC_KEY, REQ_KEY, CREDITS_KEY, URL_KEY, GEN_ED_KEY
)
from ucsc_course_assistant.course_rag import retrieve

_RAG_INDEX = None


def set_index(rag_index):
    global _RAG_INDEX
    _RAG_INDEX = rag_index

def get_index():
    if _RAG_INDEX is None:
        raise RuntimeError("Course index is not initialized.")
    return _RAG_INDEX

@tool
def search_courses(query: str, k: int = 5) -> str:
    """
    Search UCSC courses by topic, course name, or natural language query.
    """
    results = retrieve(query, get_index(), num_results=k)
    return json.dumps(
        [
            {
                'code': r['course'][CODE_KEY],
                'title': r['course'][TITLE_KEY],
                'description': r['course'][DESC_KEY],
                'requirements': r['course'].get(REQ_KEY),
                'credits': r['course'].get(CREDITS_KEY),
                'url': r['course'][URL_KEY],
                'general_education_code': r['course'].get(GEN_ED_KEY),
                'score': r['score'],
            }
            for r in results
        ],
        ensure_ascii=False,
    )


def get_tools():
    return (
        search_courses,
    )