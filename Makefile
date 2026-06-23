.PHONY: install test outputs clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest

outputs:
	python scripts/run_experiments.py

clean:
	rm -rf .pytest_cache build dist src/*.egg-info src/options_lab/__pycache__ tests/__pycache__
