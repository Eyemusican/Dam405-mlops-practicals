install:
	pip install -r requirements.txt
lint:
	flake8 src tests
test:
	python -m pytest -q
all: install lint test