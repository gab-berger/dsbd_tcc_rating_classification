import os
import sys
import subprocess
import ollama
import pandas as pd
import hashlib

def create_venv(venv_dir="venv"):
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
    print(f"Virtual environment created at: {venv_dir}")

def install_requirements(venv_dir="venv"):
    pip_path = os.path.join(venv_dir, "bin", "pip")
    if os.path.exists("requirements.txt"):
        print("Installing dependencies from requirements.txt...")
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("Dependencies installed successfully.")
    else:
        print("requirements.txt not found.")

def setup_ollama():
    print("Installing or updating Ollama...")
    try:
        subprocess.run(
            "curl -fsSL https://ollama.com/install.sh | sh",
            shell=True,
            check=True
        )
        print("Ollama installed or updated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during Ollama installation: {e}")

def download_models(models):
    for model in models:
        ollama.pull(model)
        print(f'{model} good to go!')

def download_dataset():
    download_url = 'https://www.kaggle.com/api/v1/datasets/download/davidgauthier/glassdoor-job-reviews-2'
    commands = [
        ['curl','-L','-o','data/download.zip',download_url],
        ['unzip','-o','data/download.zip'],
        ['rm','-f','data/download.zip']
    ]
    for command in commands:
        subprocess.run(command)
    print('all_reviews.csv downloaded!')

def create_comments_parquet(csv_path, parquet_path):
    """Main function to process the CSV file and save it as a Parquet file with unique IDs."""
    def load_csv(csv_path):
        """Loads the CSV file and selects the required columns."""
        print("Loading the CSV file...")
        df = pd.read_csv(csv_path, usecols=["rating", "pros", "cons", "date"])
        print("CSV file loaded successfully!")
        return df

    def clean_data(df):
        """Removes null values and duplicates, sorts the data by date, and drops the date column."""
        print("Checking for null values...")
        print(df.isnull().sum())

        print("Checking for duplicate rows...")
        print(df.duplicated().sum())

        # Remove rows with null values and duplicates
        df = df.dropna()
        df = df.drop_duplicates()
        print(f"Data after cleaning: {len(df)} rows remaining.")

        # Convert 'date' column to datetime and sort by most recent
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date', ascending=False)

        # Select the required columns, excluding the date
        df = df[["rating", "pros", "cons"]]
        print("Data sorted by date (from most recent to oldest). 'date' column removed.")
        return df

    def generate_unique_ids(df):
        """Generates unique IDs for each row based on the 'rating', 'pros', and 'cons' columns."""
        print("Generating unique IDs...")

        def generate_id(row):
            unique_string = f"{row['rating']}_{row['pros']}_{row['cons']}"
            return hashlib.sha256(unique_string.encode()).hexdigest()

        # Apply the ID generation function to each row
        df["id"] = df.apply(generate_id, axis=1)

        # Check if IDs are unique
        if df["id"].is_unique:
            print("Unique IDs generated successfully!")
        else:
            raise ValueError("ID conflicts detected!")

        return df

    def save_to_parquet(df, parquet_path):
        """Saves the DataFrame to a Parquet file."""
        print(f"Saving data to the Parquet file: {parquet_path}...")
        df.to_parquet(parquet_path, index=False)
        print("Data saved successfully!")
    
    df = load_csv(csv_path)
    df = clean_data(df)
    df = generate_unique_ids(df)
    save_to_parquet(df, parquet_path)
    print("Process completed successfully!")

def main(models):
    VENV_DIR = "venv"
    CSV_PATH = "data/all_reviews.csv"
    PARQUET_PATH = "data/test_comments.parquet"

    create_venv(VENV_DIR)
    install_requirements(VENV_DIR)
    setup_ollama()
    download_models(models)
    download_dataset()
    create_comments_parquet(CSV_PATH, PARQUET_PATH)

if __name__ == '__main__':
    models = [
        'deepseek-r1:1.5b',
        'stablelm2',
        'llama3.1',
        'llama3.2',
        'deepseek-r1:8b',
        'stablelm2:12b',
        'llama2:7b',
        'llama2:13b'
    ]
    main(models)