import pandas as pd
import sqlite3
import hashlib

CSV_PATH = "all_reviews.csv"  # Path to the CSV file
DB_PATH = "comments.db"       # SQLite database file name
TABLE_NAME = "comments"       # Table name in the database

def load_csv(csv_path):
    """Load the CSV file and select necessary columns."""
    print("Loading the CSV...")
    df = pd.read_csv(csv_path, usecols=["rating", "pros", "cons"])
    print("CSV loaded successfully!")
    return df


def clean_data(df):
    """Remove null values and duplicates."""
    print("Checking for null values...")
    print(df.isnull().sum())
    print("Checking for duplicated values...")
    print(df.duplicated().sum())
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"Data after cleaning: {len(df)} rows remaining.")
    return df


def generate_unique_ids(df):
    """Generate unique IDs for each row."""
    def generate_id(row):
        unique_string = f"{row['rating']}_{row['pros']}_{row['cons']}"
        return hashlib.sha256(unique_string.encode()).hexdigest()

    print("Generating unique IDs...")
    df["id"] = df.apply(generate_id, axis=1)

    if df["id"].is_unique:
        print("Unique IDs generated successfully!")
    else:
        raise ValueError("ID conflicts detected!")
    return df


def save_to_database(df, conn, table_name):
    """Save the DataFrame to a SQLite database with 'id' as the primary key."""
    print("Connecting to the SQLite database...")

    print(f"Saving data to the '{table_name}' table in the database...")
    with conn:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(f"""
        CREATE TABLE {table_name} (
            id TEXT PRIMARY KEY,
            rating INTEGER,
            pros TEXT,
            cons TEXT
        )
        """)
        df.to_sql(table_name, conn, if_exists="append", index=False)

    print("Data saved successfully!")


def verify_table_structure(conn, table_name):
    """Verify the table structure in the database."""
    print("Verifying the table structure...")
    query = f"PRAGMA table_info({table_name})"
    table_structure = pd.read_sql(query, conn)
    print("Table structure:")
    print(table_structure)


def test_query(conn, table_name):
    """Perform a test query on the database."""
    print("Performing a test query...")
    query = f"SELECT * FROM {table_name} LIMIT 5"
    df = pd.read_sql(query, conn)
    print("First rows in the table:")
    print(df)


def main():
    """Main function to execute the workflow."""
    df = load_csv(CSV_PATH)
    df = clean_data(df)
    df = generate_unique_ids(df)

    conn = sqlite3.connect(DB_PATH)
    save_to_database(df, conn, TABLE_NAME)
    verify_table_structure(conn, TABLE_NAME)
    test_query(conn, TABLE_NAME)
    
    conn.close()
    print("Process completed!")


if __name__ == "__main__":
    main()
