from __future__ import annotations
from .graph import run
from .observability import log


def main() -> None:
    print("Multi-Agent Research Assistant on NVIDIA NIM. Type a question (or 'quit').\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"quit", "exit"}:
            break

        result = run(question)
        print(f"\n{result['answer']}\n")
        if result["sources"]:
            print(f"sources: {', '.join(s for s in result['sources'] if s)}")
        if result["revisions"] > 1:
            log.info("answer revised %d time(s)", result["revisions"] - 1)
        print()


if __name__ == "__main__":
    main()
