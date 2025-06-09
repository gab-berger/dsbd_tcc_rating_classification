import os
import pandas as pd
import warnings
import time

warnings.filterwarnings("ignore")

COMMENTS_PATH = "data/comments_sample.parquet"
RATING_PRED_PATH = "data/manual_predictions.parquet"

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
        return pd.read_parquet(path).sample(frac=1).reset_index(drop=True)
    else:
        raise FileNotFoundError(f"Arquivo {path} não encontrado. Crie-o antes de continuar.")

def get_comments_to_predict(comments_df, existing_predictions_df):
    existing_ids = existing_predictions_df['id'].tolist()
    comments_df = comments_df[~comments_df['id'].isin(existing_ids)]
    return comments_df

def choose_language():
    while True:
        choice = input("Deseja avaliar comentários em (1) inglês ou (2) português? Responda utilizando o número respectivo. ").strip()
        if choice == '1':
            return 'en'
        if choice == '2':
            return 'pt'
        print("Opção inválida. Digite 1 para inglês ou 2 para português.")

def get_user_rating(comment):
    """
    Exibe o comentário para avaliação e solicita ao usuário um rating (1 a 5) ou 'q' para sair.
    Retorna um dicionário com os dados da previsão ou None se o usuário optar por sair.
    """
    print(f"""[Pros]:
{comment['pros']}
{'-'*45}
[Cons]:
{comment['cons']}
{'='*60}
Digite um número de 1 a 5 para avaliar este comentário ou 'q' para sair:
""")
    
    start_time = pd.Timestamp.now()
    user_input = input().strip()
    prediction_time = (pd.Timestamp.now() - start_time).total_seconds()
    
    if user_input.lower() == 'q':
        print("Encerrando a avaliação.")
        return None
    
    try:
        rating_prediction = int(user_input)
        if rating_prediction < 1 or rating_prediction > 5:
            print("Valor inválido! Por favor, digite um número entre 1 e 5.")
            return get_user_rating(comment)
    except ValueError:
        print("Entrada inválida! Por favor, digite um número entre 1 e 5 ou 'q' para sair.")
        return get_user_rating(comment)
    
    return {
        "id": comment["id"],
        "rating": rating_prediction,
        "prediction_time": prediction_time,
        "ts_prediction": pd.Timestamp.now()
    }

def safe_save_to_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Safely write DataFrame `df` to Parquet at `path` with minimal dependencies:
    1. Rename existing file to a backup (.bak) if present
    2. Write new DataFrame to a temporary file (.tmp)
    3. Read back and verify row count matches
    4. Atomically replace original with the temp file
    """
    
    if os.path.exists(path):
        bak_path = f"{path}.bak"
        os.replace(path, bak_path)

    tmp_path = f"{path}.tmp"
    df.to_parquet(tmp_path)

    df_check = pd.read_parquet(tmp_path)
    if len(df_check) != len(df):
        raise ValueError(
            f"Integrity check failed: read {len(df_check)} rows vs {len(df)} expected"
        )

    os.replace(tmp_path, path)

def save_updated_predictions(new_predictions, existing_predictions_df):
    new_predictions_df = pd.DataFrame(new_predictions)
    updated_predictions_df = pd.concat([existing_predictions_df, new_predictions_df], ignore_index=True)
    safe_save_to_parquet(updated_predictions_df, RATING_PRED_PATH)
    print(f"Avaliações salvas em {RATING_PRED_PATH}")

def main(loop_interval):
    comments_df = load_comments(COMMENTS_PATH)
    
    language = choose_language()
    comments_df = comments_df[comments_df['language']==language]

    existing_predictions_df = load_manual_predictions(RATING_PRED_PATH)
    comments_df = get_comments_to_predict(comments_df, existing_predictions_df)
    
    new_predictions_list = []
    iteration = 0

    while True:
        updated_predictions = pd.concat([existing_predictions_df, pd.DataFrame(new_predictions_list)], ignore_index=True)
        eligible_comments = get_comments_to_predict(comments_df, updated_predictions)
        
        if eligible_comments.empty:
            print("Não há mais comentários para avaliar.")
            break
        
        print(f"{'='*27}[{iteration+1}/{loop_interval}]{'='*27}")
        user_prediction = get_user_rating(eligible_comments.iloc[0])
        
        if user_prediction is None:
            break
        
        new_predictions_list.append(user_prediction)
        iteration += 1
        print("Avaliação registrada.")
        
        if iteration % loop_interval == 0:
            if new_predictions_list:
                save_updated_predictions(new_predictions_list, existing_predictions_df)
                new_predictions_list = []
            start_time = time.time()
            comments_df = get_comments_to_predict(comments_df, existing_predictions_df)
            print(f"Comentários avaliados e ordenados em {time.time() - start_time:.2f}s")

    if new_predictions_list:
        save_updated_predictions(new_predictions_list, existing_predictions_df)
    
    print("Processo concluído!")

if __name__ == "__main__":
    LOOP_INTERVAL = 10
    main(LOOP_INTERVAL)
