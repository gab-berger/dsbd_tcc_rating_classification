import os
import sys
import subprocess
import ollama

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

def download_dataset():
    download_url = 'https://www.kaggle.com/api/v1/datasets/download/davidgauthier/glassdoor-job-reviews-2'
    commands = [
        ['curl','-L','-o','download.zip',download_url],
        ['unzip','-o','download.zip'],
        ['rm','-f','download.zip']
    ]
    for command in commands:
        subprocess.run(command)
    print('all_reviews.csv downloaded!')

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

def main(models):
    VENV_DIR = "venv"
    create_venv(VENV_DIR)
    install_requirements(VENV_DIR)
    setup_ollama()
    download_models(models)
    download_dataset()

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