import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

SAVE_INTERVAL = 10

# Caminhos dos arquivos
COMMENTS_PATH = "data/comments.parquet"               # Arquivo contendo os comentários
RATING_PRED_PATH = "data/pred_manual.parquet"     # Arquivo que armazenará as avaliações do usuário

def load_manual_predictions(path):
    """
    Carrega o arquivo 'pred_manual.parquet' se ele existir;
    caso contrário, cria um DataFrame vazio com as colunas necessárias.
    """
    if os.path.exists(path):
        print(f"Carregando arquivo existente: {path}")
        df = pd.read_parquet(path)
    else:
        print(f"Arquivo {path} não encontrado. Criando um novo DataFrame vazio.")
        df = pd.DataFrame(columns=["id", "rating", "ts_prediction", "prediction_time"])
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

def select_eligible_comments(comments_df, existing_predictions):
    """
    Filtra os comentários que ainda não foram avaliados, ou seja,
    que não possuem seu 'id' presente na coluna 'id' do existing_predictions.
    """
    if existing_predictions.empty:
        return comments_df
    rated_ids = set(existing_predictions["id"])
    return comments_df[~comments_df["id"].isin(rated_ids)]

def main():
    comments_df = load_comments(COMMENTS_PATH)
    existing_predictions = load_manual_predictions(RATING_PRED_PATH)
    
    new_predictions = []
    iteration = 0

    while True:
        updated_predictions = pd.concat([existing_predictions, pd.DataFrame(new_predictions)], ignore_index=True)
        eligible_comments = select_eligible_comments(comments_df, updated_predictions)
        
        if eligible_comments.empty:
            print("Não há mais comentários para avaliar.")
            break
        
        comment = eligible_comments.iloc[0]
        
        print("\n------------------------------")
        print("Comentário para avaliação:")
        print(f"Pros: {comment['pros']}")
        print(f"Cons: {comment['cons']}")
        print("------------------------------")
        print("Digite um número de 1 a 5 para avaliar este comentário ou 'q' para sair:")
        
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
        new_predictions.append({
            "id": comment["id"],
            "rating": rating_value,
            "prediction_time": prediction_time,
            "ts_prediction": ts_now
        })
        iteration += 1
        print("Avaliação registrada.")
        
        if iteration % SAVE_INTERVAL == 0:
            if new_predictions:
                temp_df = pd.DataFrame(new_predictions)
                existing_predictions = pd.concat([existing_predictions, temp_df], ignore_index=True)
                existing_predictions.to_parquet(RATING_PRED_PATH, index=False)
                print(f"{iteration} avaliações salvas em {RATING_PRED_PATH}.")
                new_predictions = []

    if new_predictions:
        temp_df = pd.DataFrame(new_predictions)
        existing_predictions = pd.concat([existing_predictions, temp_df], ignore_index=True)
        existing_predictions.to_parquet(RATING_PRED_PATH, index=False)
        print("Avaliações finais salvas.")
    
    print("Processo concluído!")

if __name__ == "__main__":
    main()
