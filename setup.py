import subprocess

def download_dataset():
    download_url = 'https://www.kaggle.com/api/v1/datasets/download/manishkr1754/capgemini-employee-reviews-dataset'
    commands = [
        ['curl','-L','-o','download.zip',download_url],
        ['unzip','-o','download.zip'],
        ['rm','-f','download.zip']
    ]
    for command in commands:
        subprocess.run(command)
    print('Capgemini_Employee_Reviews_from_AmbitionBox.csv downloaded')

def download_models(models):
    # Install ollama with: curl -fsSL https://ollama.com/install.sh | sh
    def model_not_downloaded(model_name):
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
            return model_name not in result.stdout
        except subprocess.CalledProcessError:
            return True
    
    for model in models:
        if model_not_downloaded(model):
            subprocess.run(["ollama","pull",model])

if __name__ == '__main__':
    models = [
        'stablelm2',
        'llama3.2',
        'llama3.1',
        'stablelm2:12b'
    ]
    
    download_models(models)
    download_dataset()