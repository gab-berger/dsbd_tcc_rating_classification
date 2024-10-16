import subprocess
from time import time

def main():
    pergunta = 'O que são LLMs?'
    comando = ['ollama', 'run', 'llama3.2']
    start_time = time()
    try:
        resultado = subprocess.run(comando, input=pergunta, capture_output=True, text=True)
        if resultado.returncode != 0:
            print(f"Ocorreu um erro ao executar o Ollama:\n{resultado.stderr}")
        else:
            print(resultado.stdout)
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
    print('Tempo de execução:',round(time() - start_time,1),'segundos')

if __name__ == '__main__':
    main()
