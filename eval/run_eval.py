"""Evaluation harness: run the multi-agent system over a dataset and grade with LLM-as-judge.

    python -m eval.run_eval
"""
from __future__ import annotations
import json
import pathlib
import time
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.graph import run
from src.llm import get_chat_model

JUDGE = ChatPromptTemplate.from_messages([
    ("system",
     "You are grading an answer against a reference. Reply with only 'correct' or 'incorrect'. "
     "The answer is correct if it conveys the same factual information, even if worded differently."),
    ("human", "Question: {q}\nReference: {ref}\nAnswer: {ans}"),
])


def main() -> None:
    judge = JUDGE | get_chat_model(temperature=0) | StrOutputParser()
    rows = [json.loads(line) for line in pathlib.Path("eval/dataset.jsonl").read_text().splitlines() if line.strip()]
    correct = 0
    latencies = []
    print(f"Running {len(rows)} eval items...\n")
    for i, row in enumerate(rows, 1):
        t0 = time.perf_counter()
        result = run(row["question"])
        latencies.append(time.perf_counter() - t0)
        verdict = judge.invoke({"q": row["question"], "ref": row["expected"], "ans": result["answer"]}).strip().lower()
        ok = verdict.startswith("correct")
        correct += ok
        print(f"[{i}] {'PASS' if ok else 'FAIL'}  {row['question']}")
    n = len(rows)
    print(f"\naccuracy: {correct}/{n} = {correct / n:.0%}")
    print(f"avg latency: {sum(latencies) / n:.1f}s")


if __name__ == "__main__":
    main()
