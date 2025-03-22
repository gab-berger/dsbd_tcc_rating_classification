import os
import pandas as pd
import warnings

# Ignora todos os avisos
warnings.filterwarnings("ignore")

# Parâmetro editável: número de comentários avaliados a cada salvamento
SAVE_INTERVAL = 10

# Caminhos dos arquivos
COMMENTS_PATH = "data/comments.parquet"               # Arquivo contendo os comentários
RATING_PRED_PATH = "data/pred_manual.parquet"     # Arquivo que armazenará as avaliações do usuário

def load_or_create_pred_manual(path):
    """
    Carrega o arquivo 'pred_manual.parquet' se ele existir;
    caso contrário, cria um DataFrame vazio com as colunas necessárias.
    """
    if os.path.exists(path):
        print(f"Carregando arquivo existente: {path}")
        df = pd.read_parquet(path)
    else:
        print(f"Arquivo {path} não encontrado. Criando um novo DataFrame vazio.")
        df = pd.DataFrame(columns=["id", "pred_manual", "ts_prediction", "prediction_time"])
    return df

def load_comments(path):
    """
    Carrega o arquivo 'comments.parquet' contendo os comentários.
    Certifique-se de que este arquivo foi criado previamente.
    """
    if os.path.exists(path):
        print(f"Carregando comentários de: {path}")
        return pd.read_parquet(path)
    else:
        raise FileNotFoundError(f"Arquivo {path} não encontrado. Crie-o antes de continuar.")

def filter_unrated_comments(comments_df, rating_pred_df):
    """
    Filtra os comentários que ainda não foram avaliados, ou seja,
    que não possuem seu 'id' presente na coluna 'id' do rating_pred_df.
    """
    if rating_pred_df.empty:
        return comments_df
    rated_ids = set(rating_pred_df["id"])
    return comments_df[~comments_df["id"].isin(rated_ids)]

def main():
    # Carrega ou cria a tabela de avaliações e carrega os comentários
    rating_pred_df = load_or_create_pred_manual(RATING_PRED_PATH)
    comments_df = load_comments(COMMENTS_PATH)
    
    new_entries = []  # Lista para armazenar as avaliações realizadas nesta sessão
    iteration = 0     # Contador de avaliações realizadas

    while True:
        # Atualiza os comentários não avaliados considerando as avaliações já salvas e as novas desta sessão
        combined_ratings = pd.concat([rating_pred_df, pd.DataFrame(new_entries)], ignore_index=True)
        unrated_comments = filter_unrated_comments(comments_df, combined_ratings)
        
        if unrated_comments.empty:
            print("Não há mais comentários para avaliar.")
            break
        
        # Seleciona o primeiro comentário não avaliado
        comment = unrated_comments.iloc[0]
        
        # Exibe o comentário para o usuário (apenas os 'pros' e 'cons')
        print("\n------------------------------")
        print("Comentário para avaliação:")
        print(f"Pros: {comment['pros']}")
        print(f"Cons: {comment['cons']}")
        print("------------------------------")
        print("Digite um número de 1 a 5 para avaliar este comentário ou 'q' para sair:")
        
        # Inicia a contagem do tempo de decisão do usuário
        start_time = pd.Timestamp.now()
        user_input = input().strip()
        prediction_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        if user_input.lower() == 'q':
            print("Encerrando a avaliação.")
            break
        
        try:
            rating_value = int(user_input)
            if rating_value < 1 or rating_value > 5:
                print("Valor inválido! Por favor, digite um número entre 1 e 5.")
                continue
        except ValueError:
            print("Entrada inválida! Por favor, digite um número entre 1 e 5 ou 'q' para sair.")
            continue
        
        ts_now = pd.Timestamp.now()
        new_entries.append({
            "id": comment["id"],
            "rating": rating_value,
            "prediction_time": prediction_time,
            "ts_prediction": ts_now
        })
        iteration += 1
        print("Avaliação registrada.")
        
        # A cada SAVE_INTERVAL avaliações, salva as novas entradas no arquivo Parquet
        if iteration % SAVE_INTERVAL == 0:
            if new_entries:
                temp_df = pd.DataFrame(new_entries)
                rating_pred_df = pd.concat([rating_pred_df, temp_df], ignore_index=True)
                rating_pred_df.to_parquet(RATING_PRED_PATH, index=False)
                print(f"{iteration} avaliações salvas em {RATING_PRED_PATH}.")
                new_entries = []  # Reseta a lista de novas entradas

    # Ao sair do loop, se houver avaliações pendentes, salve-as
    if new_entries:
        temp_df = pd.DataFrame(new_entries)
        rating_pred_df = pd.concat([rating_pred_df, temp_df], ignore_index=True)
        rating_pred_df.to_parquet(RATING_PRED_PATH, index=False)
        print("Avaliações finais salvas.")
    
    print("Processo concluído!")

if __name__ == "__main__":
    main()
