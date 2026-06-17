"""Four specialised agents:
- planner    : decomposes the user query into focused sub-questions
- retriever  : thin wrapper; retrieval handled in graph.py via get_retriever()
- answer     : drafts an answer from retrieved context
- critic     : reviews the draft; returns 'approved' or a revision request
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .llm import get_chat_model

# --- Planner -----------------------------------------------------------------

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research planner. Break the user question into 1-3 focused "
     "sub-questions that together answer it fully. Return them as a numbered list, "
     "nothing else."),
    ("human", "{question}"),
])


def planner_chain():
    return PLANNER_PROMPT | get_chat_model(temperature=0) | StrOutputParser()


# --- Answer ------------------------------------------------------------------

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research assistant. Answer the question using ONLY the provided "
     "context. Cite sources by filename. Be concise and factual. If the context "
     "does not contain the answer, say you don't have enough information."),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])


def answer_chain():
    return ANSWER_PROMPT | get_chat_model() | StrOutputParser()


# --- Critic ------------------------------------------------------------------

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict answer reviewer. Check whether the answer fully and "
     "accurately addresses the question using the provided context.\n"
     "If it is correct and complete, reply with exactly: approved\n"
     "If it needs improvement, reply with: revise: <specific instruction>"),
    ("human",
     "Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"),
])


def critic_chain():
    return CRITIC_PROMPT | get_chat_model(temperature=0) | StrOutputParser()
