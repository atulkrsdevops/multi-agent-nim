# Multi-Agent Research Assistant on NVIDIA NIM

A **multi-agent system** built with LangGraph and NVIDIA NIM where specialised
agents collaborate, debate, and self-correct to produce verified answers.

```
User query
   |
   v
[Planner]  -- breaks query into sub-tasks
   |
   v
[Retriever Agent]  -- retrieves relevant context from corpus
   |
   v
[Answer Agent]  -- drafts an answer from retrieved context
   |
   v
[Critic Agent]  -- reviews answer, requests revision if needed
   |
   v
[Final Answer]
```

Covers **NCP-AAI** (Multi-Agent Architecture, Agent Collaboration, Evaluation)
and **NCA-GENL** (LLM integration, Prompt Engineering, Deployment).

> Free to run on hosted NIM endpoints at build.nvidia.com. No GPU required.

---

## Demo

Asking a multi-part question runs all four agents in sequence:

```text
> What is the parental leave policy and how much is the meal allowance abroad?
node=planner      duration_ms=1421   subtasks=3
node=retriever    duration_ms=915    n_docs=3
node=answer       duration_ms=1112   chars=195
node=critic       duration_ms=819    verdict=approved

Parental leave is 16 weeks paid for the primary caregiver and 4 weeks for the
secondary caregiver. The meal allowance abroad is $80 per day.
sources: leave_policy.md, expense_policy.md
```

![Demo: multi-part question answered by the agent crew](docs/screenshots/Multi-agentNIM1.jpg)

Asking a complex cross-policy question triggers the critic revision loop:

```text
node=planner   duration_ms=1189   subtasks=3
node=retriever duration_ms=922    n_docs=3
node=answer    duration_ms=2354   chars=1021
node=critic    duration_ms=920    verdict=revise
node=answer    duration_ms=3290   chars=1460
node=critic    duration_ms=1214   verdict=revise
answer revised 1 time(s)
```

![Demo: critic revision loop firing on a complex query](docs/screenshots/Multi-agentNIM2.jpg)

---

## Results

The eval harness (`python -m eval.run_eval`) runs a labelled question set through
the agent crew and grades each answer with an LLM-as-judge:

```text
[1] PASS  How many days can I work remotely each week?
[2] PASS  What is the parental leave for the primary caregiver?
[3] PASS  When is a receipt required for an expense?
[4] PASS  What is the sick leave policy?
[5] PASS  What is the meal allowance for international travel?

accuracy: 5/5 = 100%
avg latency: 4.0s
```

Note: question 4 triggered a critic revision — the answer agent rewrote it,
the critic approved the second attempt. Visible in the trace as `verdict=revise`
followed by `verdict=approved`.

![Eval results: 5/5 pass, 4.0s average latency](docs/screenshots/Multi-agentNIM3.jpg)

**Performance note.** The critic adds ~700ms per review pass. For latency-sensitive
use cases, the revision budget (`MAX_CRITIC_REVISIONS`) can be set to 1 or 0 in
`.env` to trade answer quality for speed.

---

## How it differs from Project 1 (agentic-rag-nim)

| | Project 1 | This project |
|---|---|---|
| Architecture | Single CRAG agent | Multi-agent crew |
| Self-correction | Query rewriting | Critic-driven revision loop |
| Planning | None | Planner decomposes queries |
| Agent count | 1 | 4 (planner, retriever, answer, critic) |
| NCP-AAI focus | Agent dev, retrieval | Multi-agent, collaboration |

---

## How it maps to the exam blueprints

| Component | File | NCA-GENL domain | NCP-AAI domain |
|---|---|---|---|
| NIM chat client | `src/llm.py` | LLM integration | Agent development |
| NeMo Retriever + FAISS | `src/embeddings.py`, `src/retriever.py` | RAG | Retrieval pipelines |
| Planner agent | `src/agents.py` | Prompt engineering | Reasoning & planning |
| Answer agent | `src/agents.py` | LLM integration | Agent development |
| Critic agent | `src/agents.py` | Alignment | Multi-agent collaboration |
| LangGraph state machine | `src/graph.py` | -- | Agent architecture |
| Per-node tracing | `src/observability.py` | Experimentation | Observability |
| LLM-as-judge eval | `eval/run_eval.py` | Evaluation | Evaluation & tuning |
| Docker + CI | `Dockerfile`, `.github/` | Deployment | Deployment & scaling |

---

## Quickstart

```bash
# Windows
copy .env.example .env        # paste your nvapi-... key
pip install -r requirements.txt
python -m src.ingest
python -m src.app

# Mac/Linux
cp .env.example .env
make setup && make ingest && make run
```

## Project structure

```
multi-agent-nim/
├── src/
│   ├── settings.py       # env-driven config
│   ├── llm.py            # ChatNVIDIA factory
│   ├── embeddings.py     # NVIDIAEmbeddings factory
│   ├── ingest.py         # build FAISS index
│   ├── retriever.py      # load FAISS index
│   ├── agents.py         # planner, answer, critic agents
│   ├── graph.py          # LangGraph multi-agent state machine
│   ├── observability.py  # per-node tracing
│   └── app.py            # CLI entrypoint
├── data/sample_docs/     # knowledge base (swap your own)
├── eval/                 # dataset + eval runner
├── tests/                # offline smoke tests
└── .github/workflows/    # CI pipeline
```

## License

MIT
