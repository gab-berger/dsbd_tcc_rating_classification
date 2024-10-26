import pandas as pd
import subprocess
from time import time

def download_dataset():
    download_url = 'https://www.kaggle.com/api/v1/datasets/download/manishkr1754/capgemini-employee-reviews-dataset'
    commands = [
        ['curl','-L','-o','download.zip',download_url],
        ['unzip','-o','download.zip'],
        ['rm','-f','download.zip']
    ]
    for command in commands:
        subprocess.run(command)

def message_ollama(prompt, model='llama3.2'):
    command = ['ollama', 'run', model]
    start_time = time()
    
    try:
        resultado = subprocess.run(command, input=prompt, capture_output=True, text=True)
        if resultado.returncode != 0:
            print(f"Ocorreu um erro ao executar o Ollama:\n{resultado.stderr}")
        else:
            result = resultado.stdout
            total_time = round(time() - start_time,1)
            result_str = f'========================================\n{model} - {total_time}s\n========================================\n{result}'
            print(result_str)
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
    return result_str

def main(models:list, start_prompt:str, end_prompt:str)->str: 
    
    try:
        df = pd.read_csv('Capgemini_Employee_Reviews_from_AmbitionBox.csv', usecols=['Likes','Dislikes'])
    except:
        download_dataset() 
        df = pd.read_csv('Capgemini_Employee_Reviews_from_AmbitionBox.csv', usecols=['Likes','Dislikes'])
    
    likes = df['Likes'].tolist()
    dislikes = df['Dislikes'].tolist()
    n_surveys = len(likes)

    comments = []
    for n in range(n_surveys):
        comments.append(likes[n])
        comments.append(dislikes[n])
    n_comments = len(comments)

    prompt = start_prompt
    for n in range(n_comments):
        prompt = prompt+'\n'+str(comments[n])
    prompt = prompt + end_prompt
    out_txt = f'-----> PROMPT:\n\n{start_prompt}\n\n-----> [{n_comments} comentários]\n\n{end_prompt}\nTamanho do prompt: {len(prompt)}\n\n--------------------------------------------------------------------------------\n'
    print(out_txt)

    for model in models:
        result = message_ollama(prompt, model)
        out_txt = '\n'+out_txt+result

    file_name = "output.txt"
    with open(file_name, 'w', encoding='utf-8') as txt_file:
        txt_file.write(out_txt)

if __name__ == '__main__':
    models = [
    'stablelm2',
    'llama3.2',
    'llama3.1',
    'stablelm2:12b',
    ]
    
    main_prompt = 'Você é um especialista em pesquisas organizacionais e possui ótimas habilidades em interpretar estas pesquisas.\
    Sua especialidade é resumir seus comentários em relação aos tópicos mais importantes, gerando valiosos insights que ajudam o time de Pessoas / Recursos Humanos a identificar pontos positivos e negativos em relação à percepção dos colaboradores.\
    Seu objetivo é identificar os três assuntos mais comentados na pesquisa, gerando um resumo executivo que utiliza exclusivamente os dados da pesquisa e nunca utiliza informações externas.'
    start_prompt = main_prompt+' Faça um resumo executivo da pesquisa organizacional da Amazon a partir dos comentários a seguir:'
    end_prompt = 'Com base nestes comentários, faça um resumo executivo da pesquisa organizacional da Amazon. Lembre-se de que: '+main_prompt

    main(models, start_prompt, end_prompt)