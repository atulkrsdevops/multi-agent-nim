"""Multi-agent LangGraph state machine.

Flow:
    plan -> retrieve -> answer -> critique -> [approved?] -> END
                                                  |
                                            no (under budget)
                                                  v
                                             answer (revised)
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from .agents import answer_chain, critic_chain, planner_chain
from .observability import span
from .retriever import get_retriever
from .settings import get_settings


class AgentState(TypedDict):
    question: str
    sub_questions: str
    documents: list[Document]
    answer: str
    critique: str
    revisions: int
    sources: list[str]


def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs
    )


# --- nodes -------------------------------------------------------------------

def plan(state: AgentState) -> AgentState:
    with span("planner") as a:
        sub_qs = planner_chain().invoke({"question": state["question"]})
        a["subtasks"] = len([line for line in sub_qs.splitlines() if line.strip()])
    return {**state, "sub_questions": sub_qs}


def retrieve(state: AgentState) -> AgentState:
    # Retrieve against the original question for broad coverage.
    with span("retriever") as a:
        docs = get_retriever().invoke(state["question"])
        a["n_docs"] = len(docs)
    sources = list({d.metadata.get("source", "") for d in docs})
    return {**state, "documents": docs, "sources": sources}


def answer(state: AgentState) -> AgentState:
    context = _format_context(state["documents"])
    # If there's a revision instruction, append it so the model knows what to fix.
    question = state["question"]
    if state.get("critique", "").startswith("revise:"):
        question = f"{question}\n\nRevision instruction: {state['critique'][7:].strip()}"
    with span("answer") as a:
        ans = answer_chain().invoke({"question": question, "context": context})
        a["chars"] = len(ans)
    return {**state, "answer": ans}


def critique(state: AgentState) -> AgentState:
    context = _format_context(state["documents"])
    with span("critic") as a:
        verdict = critic_chain().invoke({
            "question": state["question"],
            "context": context,
            "answer": state["answer"],
        })
        a["verdict"] = "approved" if verdict.strip().lower().startswith("approved") else "revise"
    return {**state, "critique": verdict.strip(), "revisions": state["revisions"] + 1}


# --- routing -----------------------------------------------------------------

def route_critique(state: AgentState) -> str:
    s = get_settings()
    if state["critique"].lower().startswith("approved"):
        return "end"
    if state["revisions"] >= s.max_critic_revisions:
        return "end"   # revision budget exhausted
    return "answer"


# --- assembly ----------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", plan)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_node("critique", critique)

    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "critique")
    g.add_conditional_edges("critique", route_critique, {"end": END, "answer": "answer"})

    return g.compile()


def run(question: str) -> dict:
    graph = build_graph()
    final = graph.invoke({
        "question": question,
        "sub_questions": "",
        "documents": [],
        "answer": "",
        "critique": "",
        "revisions": 0,
        "sources": [],
    })
    return {
        "answer": final["answer"],
        "sources": final["sources"],
        "revisions": final["revisions"],
        "critique": final["critique"],
    }
