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
        print(f'Pulling {model}...')
        ollama.pull(model)
        print(f'{model} good to go!')

def download_dataset():
    print('Downloading all_reviews.csv...')
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
        """Carrega o arquivo CSV e seleciona as colunas necessárias."""
        print("Carregando o CSV...")
        df = pd.read_csv(csv_path, usecols=["rating", "pros", "cons","date"])
        print("CSV carregado com sucesso!")
        return df

    def clean_data(df):
        """Remove valores nulos e duplicados."""
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date',ascending=False)
        print(f"Dados ordenados pela data (mais recente a mais antigo).")
        df = df[["rating", "pros", "cons"]]
        
        print("Verificando valores nulos...")
        print(f"{df.isnull().sum()} - valores nulos")
        df = df.dropna()

        print("Verificando valores duplicados...")
        print(f"{df.duplicated().sum()} valores duplicados")
        df = df.drop_duplicates()

        print(f"Dados após a limpeza: {len(df)} linhas restantes.")

        df['rating'] = df['rating'].astype(int)
        return df

    def generate_unique_ids(df):
        """Gera IDs únicos para cada linha com base nas colunas 'rating', 'pros' e 'cons'."""
        print("Gerando IDs únicos...")
        
        def generate_id(row):
            unique_string = f"{row['rating']}_{row['pros']}_{row['cons']}"
            return hashlib.sha256(unique_string.encode()).hexdigest()
        
        df["id"] = df.apply(generate_id, axis=1)
        
        if df["id"].is_unique:
            print("IDs únicos gerados com sucesso!")
        else:
            raise ValueError("Conflitos de ID detectados!")
        
        return df[["id", "rating", "pros", "cons"]]

    def create_metrics(df):
        """Cria colunas de métricas para cada comentário"""
        print('Criando métricas...')
        df['pros_length'] = df['pros'].str.len()
        df['cons_length'] = df['cons'].str.len()
        df['comment_length'] = df['pros_length']+df['cons_length']
        df['pros_length_proportion'] = ((df['pros_length'] / df['comment_length']) * 100).astype(int)
        df = df.drop(columns=['pros_length','cons_length'])

        for metric in ['comment_length', 'pros_length_proportion']:
            quantiles = df[metric].quantile([0.2, 0.4, 0.6, 0.8])
            bins = [df[metric].min() - 1, quantiles[0.2], quantiles[0.4], quantiles[0.6], quantiles[0.8], df[metric].max() + 1]
            labels = [1, 2, 3, 4, 5]

            df[f'{metric}_group'] = pd.cut(df[metric], bins=bins, labels=labels)

        print('Métricas criadas!')
        return df

    def save_to_parquet(df, parquet_path):
        """Salva o DataFrame em um arquivo Parquet."""
        print(f"Salvando os dados no arquivo Parquet: {parquet_path}...")
        df.to_parquet(parquet_path, index=False)
        print("Dados salvos com sucesso!")

    df = load_csv(csv_path)
    df = clean_data(df)
    df = generate_unique_ids(df)
    df = create_metrics(df)
    save_to_parquet(df, parquet_path)
    print("Processo concluído!")

def main(models):
    VENV_DIR = "venv"
    CSV_PATH = "data/all_reviews.csv"
    PARQUET_PATH = "data/comments.parquet"

    # create_venv(VENV_DIR)
    # install_requirements(VENV_DIR)
    # setup_ollama()
    # download_models(models)
    # download_dataset()
    create_comments_parquet(CSV_PATH, PARQUET_PATH)

if __name__ == '__main__':
    models  = [
        'qwen2.5:0.5b',
        'deepseek-r1:1.5b',
        'qwen2.5:1.5b',
        'stablelm2:1.6b',
        'llama3.2:3b',
        'qwen2.5:3b',
        'llama2:7b',
        'deepseek-r1:8b',
        'llama3.1:8b',
        'stablelm2:12b',
        'llama2:13b'
    ]
    main(models)