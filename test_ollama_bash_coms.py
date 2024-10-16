import pandas as pd
import subprocess
from time import time

def message_ollama(prompt, model='llama3.2'):
    command = ['ollama', 'run', model]
    start_time = time()
    try:
        resultado = subprocess.run(command, input=prompt, capture_output=True, text=True)
        if resultado.returncode != 0:
            print(f"Ocorreu um erro ao executar o Ollama:\n{resultado.stderr}")
        else:
            print(command[2],'- ',round(time() - start_time,1),'segundos:\n')
            print(resultado.stdout)
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

def main(): 
    start_prompt = 'A seguir, te mando uma série de comentários de uma pesquisa organizaional, um a cada linha, e gostaria que você fizesse um resumo destes comentários:'
    end_prompt = 'Com base nestes comentários, faça um resumo, em português do Brasil, que contenha 2 parágrafos: o primeiro parágrafo resumindo os pontos positivos, e o segundo contendo os pontos negativos. Fique atento ao que foi escrito, utilize apenas isso como base, não adicione novas informações.'

    df = pd.read_csv('Amazon_Reviews.csv', usecols=['Likes','Dislikes'])
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
        prompt = prompt+'\n'+comments[n]
    prompt = prompt + end_prompt
    print(start_prompt,f'\n({n_comments} comentários)\n',end_prompt,'\n')

    message_ollama(prompt, 'llama3.2')
    message_ollama(prompt, 'llama3.1')

if __name__ == '__main__':
    main()