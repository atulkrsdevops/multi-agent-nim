from __future__ import annotations


def test_settings_load():
    from src.settings import get_settings
    s = get_settings()
    assert s.nim_base_url.startswith("https://")
    assert s.max_critic_revisions >= 1


def test_graph_compiles():
    from src.graph import build_graph
    g = build_graph()
    assert g is not None


def test_route_critique_approved():
    from src.graph import route_critique
    state = {"critique": "approved", "revisions": 1,
             "question": "", "sub_questions": "", "documents": [],
             "answer": "", "sources": []}
    assert route_critique(state) == "end"


def test_route_critique_revise():
    from src.graph import route_critique
    state = {"critique": "revise: add more detail", "revisions": 1,
             "question": "", "sub_questions": "", "documents": [],
             "answer": "", "sources": []}
    assert route_critique(state) == "answer"


def test_route_critique_budget_exhausted():
    from src.graph import route_critique
    from src.settings import get_settings
    s = get_settings()
    state = {"critique": "revise: still wrong", "revisions": s.max_critic_revisions,
             "question": "", "sub_questions": "", "documents": [],
             "answer": "", "sources": []}
    assert route_critique(state) == "end"
