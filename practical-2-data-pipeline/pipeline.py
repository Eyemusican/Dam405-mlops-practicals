import pandas as pd
import os
import sqlite3

RAW = 'data/raw'
os.makedirs(RAW, exist_ok=True)


def extract(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.to_csv(f'{RAW}/source.csv', index=False)  # keep raw copy
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna().copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    # derived feature: total family members aboard (siblings/spouses + parents/children + self)
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    return df.reset_index(drop=True)


def validate(df: pd.DataFrame):
    assert not df.empty, "DataFrame is empty"
    assert df.isnull().sum().sum() == 0, "Data still has null values"
    assert df.columns.duplicated().sum() == 0, "Duplicate columns found"
    print('Validation passed.')


def load(df: pd.DataFrame, db='data/warehouse.db', table='clean_data'):
    with sqlite3.connect(db) as conn:
        df.to_sql(table, conn, if_exists='replace', index=False)
    df.to_parquet('data/clean_data.parquet', index=False)


def run(url: str = 'data/raw/source.csv'):
    df = extract(url)
    df = transform(df)
    validate(df)
    load(df)
    print(f'Pipeline complete. Final rows: {len(df)}, Columns: {len(df.columns)}')


if __name__ == '__main__':
    run()