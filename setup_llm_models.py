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

def main(models):
    VENV_DIR = "venv"
    CSV_PATH = "data/all_reviews.csv"
    PARQUET_PATH = "data/comments.parquet"

    # create_venv(VENV_DIR)
    # install_requirements(VENV_DIR)
    # setup_ollama()
    download_models(models)

if __name__ == '__main__':
    models  = [
        'qwen2.5:0.5b',
        'qwen3:0.6b',
        'gemma3:1b',
        'deepseek-r1:1.5b',
        'qwen2.5:1.5b',
        'stablelm2:1.6b',
        'qwen3:1.7b',
        'llama3.2:3b',
        'qwen2.5:3b',
        'gemma3:4b',
        'phi3:3.8b',
        'qwen3:4b',
        'llama2:7b',
        'deepseek-r1:8b',
        'llama3.1:8b',
        'qwen3:8b',
        'stablelm2:12b',
        'gemma3:12b',
        'llama2:13b'
    ]
    main(models)