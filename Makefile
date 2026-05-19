evaluate:
	uv run python -m student search_dataset data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10
	uv run python -m student search_dataset data/datasets/AnsweredQuestions/dataset_code_public.json --k 10
	./moulinette-ubuntu evaluate_student_search_results \
		data/output/search_results/dataset_docs_public.json \
		data/datasets/AnsweredQuestions/dataset_docs_public.json \
		--k 10 --max_context_length 2000 --threshold 0.80
	./moulinette-ubuntu evaluate_student_search_results \
		data/output/search_results/dataset_code_public.json \
		data/datasets/AnsweredQuestions/dataset_code_public.json \
		--k 10 --max_context_length 2000 --threshold 0.50

index:
	uv run python -m src index data/raw/vllm-0.10.1

install:
	uv sync

run:
	uv run python -m student

debug:
	uv run python -m pdb -m student

clean:
	rm -rf __pycache__ .mypy_cache */**/__pycache__

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores \
	        --ignore-missing-imports --disallow-untyped-defs \
	        --check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict
