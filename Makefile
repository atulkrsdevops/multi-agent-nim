.PHONY: setup ingest run eval test lint

setup:
	pip install -r requirements.txt

ingest:
	python -m src.ingest

run:
	python -m src.app

eval:
	python -m eval.run_eval

test:
	pytest -q

lint:
	ruff check .
