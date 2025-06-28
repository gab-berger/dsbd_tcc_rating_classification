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
        # Qwen Family (8 models)
        'qwen2.5:0.5b',
        'qwen2.5:1.5b',
        'qwen2.5:3b',
        'qwen3:0.6b',
        'qwen3:1.7b',
        'qwen3:14b',
        'qwen3:4b',
        'qwen3:8b',

        # Gemma Family (7 models)
        'gemma:2b',
        'gemma:7b',
        'gemma3:1b',
        'gemma3:12b',
        'gemma3:4b',
        'gemma3n:e2b',
        'gemma3n:e4b',

        # DeepSeek Family (5 models)
        'deepseek-r1:1.5b',
        'deepseek-r1:14b',
        'deepseek-r1:7b',
        'deepseek-r1:7b',
        'deepseek-r1:8b',

        # Orca Family (5 models)
        'orca-mini:13b',
        'orca-mini:3b',
        'orca-mini:7b',
        'orca2:13b',
        'orca2:7b',

        # Llama Family (4 models)
        'llama2:13b',
        'llama2:7b',
        'llama3.1:8b',
        'llama3.2:3b',

        # Cogito Family (3 models)
        'cogito:14b',
        'cogito:3b',
        'cogito:8b',

        # Phi Family (3 models)
        'phi3:3.8b',
        'phi4-mini-reasoning:3.8b',
        'phi4-reasoning:14b',

        # StableLM Family (2 models)
        'stablelm2:1.6b',
        'stablelm2:12b',

        # Yi Family (2 models)
        'yi:6b',
        'yi:9b',

        # Olmo Family (2 models)
        'olmo2:13b',
        'olmo2:7b',

        # Exaone Family (2 models)
        'exaone-deep:2.4b',
        'exaone-deep:7.8b',

        # Mistral Family (2 models)
        'mistral-nemo:12b',
        'mistral-openorca:7b',

        # EverythingLM Family (1 model)
        'everythinglm:13b',

        # DeepScaler Family (1 model)
        'deepscaler:1.5b',

        # OpenThinker Family (1 model)
        'openthinker:7b',
    ]
    main(models)