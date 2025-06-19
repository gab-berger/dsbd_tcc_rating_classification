import pandas as pd

def gen_analytic_df():
    df_llm_path = 'data/llm_predictions.parquet'
    df_manual_path = 'data/manual_predictions.parquet'
    df_comments_path = 'data/comments_sample.parquet'

    df_llm = pd.read_parquet(
        df_llm_path,
        columns=['id','model','rating','processing_time']
        ).rename(
            columns={'rating':'model_rating'}
        )
    df_manual = pd.read_parquet(
        df_manual_path,
        columns=['id','rating']
        ).rename(
            columns={'rating':'manual_rating'}
        )
    df_comments = pd.read_parquet(
        df_comments_path,
        columns=['id','rating','language']
        ).rename(
            columns={'rating':'real_rating'}
        )

    df = (
        df_manual
        .merge(df_llm, on='id', how='left')
        .merge(df_comments, on='id', how='inner')
        )

    for type in ['manual','model']:
        df[f'is_inconsistent_{type}'] = (
            (df[f'{type}_rating'] - df['real_rating']).abs() > 1
            ).astype(int)
        
    print(df.info())
    return df

if __name__ == '__main__':
    PATH = 'data/analytic_df.parquet'
    
    df = gen_analytic_df()
    df.to_parquet(PATH)
    print(f"Saved df to '{PATH}'")