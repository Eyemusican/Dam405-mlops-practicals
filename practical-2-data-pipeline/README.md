# Practical 2 — Data Pipeline (ETL)

A small ETL (Extract, Transform, Load) pipeline built in Python. It pulls the
Titanic dataset from a public CSV source, cleans it, validates it, and loads
the curated data into both a SQLite database and a Parquet file.

## What this project does

- Extracts the Titanic dataset and keeps an untouched raw copy
- Cleans the data (drops missing values, renames columns) and adds a
  derived feature, family_size
- Validates the cleaned data (not empty, no nulls, no duplicate columns)
- Loads the curated data into a SQLite table (clean_data) and a Parquet file

## Project structure

```

practical-2-data-pipeline/
├── pipeline.py
└── data/
├── raw/source.csv # untouched raw copy
├── warehouse.db # SQLite database
└── clean_data.parquet # Parquet file

```

## Setup

1. Reuse the virtual environment from Practical 1 (or create a new one):
   python -m venv .venv
   .venv\Scripts\Activate.ps1   (Windows)
2. Install dependencies:
   pip install pandas requests pyarrow

## Running the pipeline

Run the whole pipeline (extract, transform, validate, load) with one command:

    python pipeline.py

## Confirming the output

Query the SQLite database directly to check the data landed correctly:

    python -c "import sqlite3, pandas as pd; conn = sqlite3.connect('data/warehouse.db'); print(pd.read_sql('SELECT * FROM clean_data LIMIT 5', conn))"