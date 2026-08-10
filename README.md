# MLOps Lab 1 — Basic MLOps Environment

A small MLOps project demonstrating a reproducible Python environment, automated
testing, linting, and continuous integration using GitHub Actions.

## What this project does

Trains a Logistic Regression model on the Iris dataset and checks that it
achieves at least 80% accuracy, using a standard, testable project structure.

## Project structure

```
mlops-lab1/
├── src/            # source code (model.py)
├── tests/          # tests (test_model.py)
├── requirements.txt
├── Makefile
└── .github/workflows/ci.yml
```

## Setup

1. Clone this repository
2. Create and activate a virtual environment:
   python -m venv .venv
   .venv\Scripts\Activate.ps1   (Windows)
3. Install dependencies:
   pip install -r requirements.txt

## Running the project

Run everything (install, lint, test) in one command:

    make all

Or run steps individually:

    make install   # install dependencies
    make lint      # check code style with flake8
    make test      # run tests with pytest

## Continuous Integration

Every push automatically runs the same install, lint, and test steps via
GitHub Actions (see `.github/workflows/ci.yml`).