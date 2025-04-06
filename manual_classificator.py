import os
import pandas as pd
import warnings
import time

warnings.filterwarnings("ignore")

COMMENTS_PATH = "data/comments.parquet"
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

def select_eligible_comments(comments_df, existing_predictions):
    """
    Filtra os comentários que ainda não foram avaliados, ou seja,
    que não possuem seu 'id' presente na coluna 'id' do existing_predictions.
    """
    if existing_predictions.empty:
        return comments_df
    rated_ids = set(existing_predictions["id"])
    return comments_df[~comments_df["id"].isin(rated_ids)]

def sort_top_score_comments(comments_df, existing_predictions):
    """
    Seleciona os 5 comentários elegíveis com maior score, priorizando grupos sub-representados
    nas métricas 'rating', 'comment_length_group' e 'pros_length_proportion_group'.
    A hierarquia de importância é: rating > comment_length_group > pros_length_proportion_group.
    """
    initial_cols = list(comments_df.columns)
    existing_predictions_metrics = existing_predictions.merge(
        comments_df[['id', 'comment_length_group', 'pros_length_proportion_group']],
        on='id',
        how='inner'
    )
    metrics = ['rating', 'comment_length_group', 'pros_length_proportion_group']
    
    dfs = {
    col: (
        existing_predictions_metrics[col]
        .value_counts()
        .rename('count')
        .sort_values(ascending=True)
        .reset_index()
        .rename(columns={'index': col})
    )
    for col in metrics
    }

    for col in metrics:
        df_metric = dfs[col]
        df_metric['ranking'] = df_metric['count'].rank(method='min', ascending=True).astype(int)
        max_rank = df_metric['ranking'].max()
        base_score = (max_rank - df_metric['ranking'] + 1).astype(int)
        multiplier = 100 if col == 'rating' else 10 if col == 'comment_length_group' else 1
        df_metric[f'{col}_score'] = base_score * multiplier
        dfs[col] = df_metric
    
    top_score_comments = comments_df.copy()
    top_score_comments = top_score_comments.merge(
        dfs['rating'][['rating', 'rating_score']], on='rating', how='left'
    ).merge(
        dfs['comment_length_group'][['comment_length_group', 'comment_length_group_score']], on='comment_length_group', how='left'
    ).merge(
        dfs['pros_length_proportion_group'][['pros_length_proportion_group', 'pros_length_proportion_group_score']],
        on='pros_length_proportion_group', how='left'
    )
    
    top_score_comments['score'] = (
        top_score_comments['rating_score'].fillna(0) +
        top_score_comments['comment_length_group_score'].fillna(0) +
        top_score_comments['pros_length_proportion_group_score'].fillna(0)
    )
    
    top_score_comments = top_score_comments.sort_values(by='score', ascending=False)
    return top_score_comments[initial_cols]

def get_user_rating(comment):
    """
    Exibe o comentário para avaliação e solicita ao usuário um rating (1 a 5) ou 'q' para sair.
    Retorna um dicionário com os dados da previsão ou None se o usuário optar por sair.
    """
    print(f"""
{'='*60}
[Pros]: {comment['pros']}
{'-'*30}
[Cons]: {comment['cons']}
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

def save_updated_predictions(new_predictions, existing_predictions):
    new_predictions = pd.DataFrame(new_predictions)
    updated_predictions = pd.concat([existing_predictions, new_predictions], ignore_index=True)
    updated_predictions.to_parquet(RATING_PRED_PATH, index=False)
    print(f"Avaliações salvas em {RATING_PRED_PATH}")

def main(loop_interval):
    comments_df = load_comments(COMMENTS_PATH)
    existing_predictions = load_manual_predictions(RATING_PRED_PATH)
    comments_df = sort_top_score_comments(comments_df, existing_predictions)
    
    new_predictions = []
    iteration = 0

    while True:
        updated_predictions = pd.concat([existing_predictions, pd.DataFrame(new_predictions)], ignore_index=True)
        eligible_comments = select_eligible_comments(comments_df, updated_predictions)
        
        if eligible_comments.empty:
            print("Não há mais comentários para avaliar.")
            break
        
        user_prediction = get_user_rating(eligible_comments.iloc[0])
        
        if user_prediction is None:
            break
        
        new_predictions.append(user_prediction)
        iteration += 1
        print("Avaliação registrada.")
        
        if iteration % loop_interval == 0:
            if new_predictions:
                save_updated_predictions(new_predictions, existing_predictions)
                new_predictions = []
            start_time = time.time()
            comments_df = sort_top_score_comments(comments_df, existing_predictions)
            print(f"Comentários avaliados e ordenados em {time.time() - start_time:.2f}s")

    if new_predictions:
        save_updated_predictions(new_predictions, existing_predictions)
    
    print("Processo concluído!")

if __name__ == "__main__":
    LOOP_INTERVAL = 10
    main(LOOP_INTERVAL)
