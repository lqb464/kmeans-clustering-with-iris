.PHONY: install notebooks validate train test lint api frontend
install:
	python -m pip install -e ".[api,frontend,notebooks,dev]"
notebooks:
	jupyter lab notebooks
validate:
	python scripts/validate_data.py
train:
	python scripts/train.py
test:
	python -m pytest
lint:
	ruff check .
api:
	uvicorn backend.app:app --reload
frontend:
	streamlit run frontend/app.py
