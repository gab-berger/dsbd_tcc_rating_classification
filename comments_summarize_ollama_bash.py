import pandas as pd
import random
import subprocess
from time import time

def generate_comments_list(comment_columns:list) -> list:
    csv_file = 'Capgemini_Employee_Reviews_from_AmbitionBox.csv'
    df = pd.read_csv(csv_file, usecols=comment_columns)
    
    comments = []
    for column in comment_columns:
        comments_column = df[df[column].notna() & (df[column] != '')][column].tolist()
        for comment in range(len(comments_column)):
            comments.append(comments_column[comment])

    random.seed(42)
    random.shuffle(comments)

    return comments

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
            result_str = f"{'=' * 40}\n{model} - {total_time}s\n{'=' * 40}\n{result}"
            print(result_str)
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
    return result_str

def main(models:list, start_prompt:str, end_prompt:str)->str: 
    comments = generate_comments_list(['Likes','Dislikes'])
    comments_n = len(comments)

    prompt = start_prompt
    for n in range(comments_n):
        prompt = prompt+'\n'+str(comments[n])
    prompt = prompt + end_prompt
    out_txt = f"-----> PROMPT:\n\n{start_prompt}\n\n-----> [{comments_n} comentários]\n\n{end_prompt}\nTamanho do prompt: {len(prompt)}\n\n{'-' * 40}\n"
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