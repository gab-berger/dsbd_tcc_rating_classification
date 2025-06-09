import subprocess
import pandas as pd
import hashlib

CSV_EN_PATH = "data/all_reviews.csv"
CSV_PT_PATH = "data/comentarios_glassdor_ajustado.csv"
MANUAL_PREDICTIONS_PATH = "data/manual_predictions.parquet"
PARQUET_PATH = "data/comments_sample.parquet"

def download_dataset():
    print('Downloading all_reviews.csv...')
    download_url = 'https://www.kaggle.com/api/v1/datasets/download/davidgauthier/glassdoor-job-reviews-2'
    commands = [
        ['curl','-L','-o','data/download.zip',download_url],
        ['unzip','-o','data/download.zip'],
        ['rm','-f','data/download.zip']
    ]
    for command in commands:
        subprocess.run(command)
    print('all_reviews.csv downloaded!')

def create_comments_df(csv_path):
    """Main function to process the CSV file and save it as a Parquet file with unique IDs."""

    def load_csv(csv_path):
        """Carrega o arquivo CSV e seleciona as colunas necessárias."""
        print(f"Carregando {csv_path} ...")
        if csv_path == CSV_EN_PATH:
            df = pd.read_csv(csv_path, usecols=["rating", "pros", "cons"])
            df['language'] = 'en'
        if csv_path == CSV_PT_PATH:
            df = pd.read_csv(csv_path, usecols=["Nota", "Pros", "Cons"])
            df = df.rename(columns={"Nota":"rating", "Pros": "pros", "Cons": "cons"})
            df['language'] = 'pt'
        print("CSV carregado com sucesso!")
        return df

    def clean_data(df):
        """Remove valores nulos e duplicados."""
        
        print("Verificando valores nulos...")
        print(f"{df.isnull().sum()} - valores nulos")
        df = df.dropna()

        print("Verificando valores duplicados...")
        print(f"{df.duplicated().sum()} valores duplicados")
        df = df.drop_duplicates()

        print(f"Dados após a limpeza: {len(df)} linhas restantes.")

        df['rating'] = df['rating'].astype(int)
        return df

    def generate_unique_ids(df):
        """Gera IDs únicos para cada linha com base nas colunas 'rating', 'pros' e 'cons'."""
        print("Gerando IDs únicos...")
        
        def generate_id(row):
            unique_string = f"{float(row['rating'])}_{row['pros']}_{row['cons']}" # float conversion to preserve initial format
            return hashlib.sha256(unique_string.encode()).hexdigest()
        
        df["id"] = df.apply(generate_id, axis=1)
        
        if df["id"].is_unique:
            print("IDs únicos gerados com sucesso!")
        else:
            raise ValueError("Conflitos de ID detectados!")
        
        return df[["id", "rating", "pros", "cons", "language"]]

    def create_metrics(df):
        """Cria colunas de métricas para cada comentário"""
        print('Criando métricas...')
        df['comment_length'] = df['pros'].str.len()+df['cons'].str.len()

        quantiles = df['comment_length'].quantile([0.2, 0.4, 0.6, 0.8])
        bins = [df['comment_length'].min() - 1, quantiles[0.2], quantiles[0.4], quantiles[0.6], quantiles[0.8], df['comment_length'].max() + 1]
        labels = [1, 2, 3, 4, 5]

        df['comment_length_group'] = pd.cut(df['comment_length'], bins=bins, labels=labels).astype(int)

        print('Métricas criadas!')
        return df

    df = load_csv(csv_path)
    df = clean_data(df)
    df = generate_unique_ids(df)
    df = create_metrics(df)
    print("Processo concluído!")
    return df

def create_sample_df(df_en, df_pt, existing_predictions_path):
    SAMPLE_SIZE = 500
    ratings = list(range(1, 6))
    lengths = list(range(1, 6))
    
    existing_ids = pd.read_parquet(existing_predictions_path, columns=['id'])['id'].tolist()
    samples = []
    
    for df_lang in [df_en, df_pt]:
        df_pred = df_lang[df_lang['id'].isin(existing_ids)]
        df_rem = df_lang[~df_lang['id'].isin(existing_ids)]
        
        desired_per_rating = SAMPLE_SIZE // len(ratings)
        desired = {}
        for r in ratings:
            per_length = desired_per_rating // len(lengths)
            for g in lengths:
                desired[(r, g)] = per_length
        
        to_concat = [df_pred]
        pred_counts = df_pred.groupby(['rating', 'comment_length_group']).size().to_dict()
        
        for (r, g), target in desired.items():
            have = pred_counts.get((r, g), 0)
            need = target - have
            if need > 0:
                pool = df_rem[(df_rem['rating'] == r) & (df_rem['comment_length_group'] == g)]
                if not pool.empty:
                    to_concat.append(pool.sample(n=min(need, len(pool)), random_state=42))
        
        df_sample = pd.concat(to_concat, ignore_index=True)
        
        if len(df_sample) < SAMPLE_SIZE:
            deficit = SAMPLE_SIZE - len(df_sample)
            used_ids = df_sample['id'].tolist()
            extra_pool = df_rem[~df_rem['id'].isin(used_ids)]
            df_sample = pd.concat([
                df_sample,
                extra_pool.sample(n=min(deficit, len(extra_pool)), random_state=42)
            ], ignore_index=True)
        elif len(df_sample) > SAMPLE_SIZE:
            df_sample = df_sample.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        
        samples.append(df_sample)
    
    return pd.concat(samples, ignore_index=True)

def save_to_parquet(df, parquet_path):
    """Salva o DataFrame em um arquivo Parquet."""
    print(f"Salvando os dados no arquivo Parquet: {parquet_path}...")
    df.to_parquet(parquet_path, index=False)
    print("Dados salvos com sucesso!")

def main():
    # download_dataset()
    df_en = create_comments_df(CSV_EN_PATH)
    df_pt = create_comments_df(CSV_PT_PATH)
    df = create_sample_df(df_en, df_pt, MANUAL_PREDICTIONS_PATH)
    save_to_parquet(df, PARQUET_PATH)

if __name__ == '__main__':
    main()