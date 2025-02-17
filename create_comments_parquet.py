import pandas as pd
import hashlib

CSV_PATH = "data/all_reviews.csv"       # Caminho para o arquivo CSV
PARQUET_PATH = "/comments.parquet"  # Caminho para o arquivo Parquet

def load_csv(csv_path):
    """Carrega o arquivo CSV e seleciona as colunas necessárias."""
    print("Carregando o CSV...")
    df = pd.read_csv(csv_path, usecols=["rating", "pros", "cons","date"])
    print("CSV carregado com sucesso!")
    return df

def clean_data(df):
    """Remove valores nulos e duplicados."""
    print("Verificando valores nulos...")
    print(df.isnull().sum())
    print("Verificando valores duplicados...")
    print(df.duplicated().sum())
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"Dados após a limpeza: {len(df)} linhas restantes.")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date',ascending=False)
    df = df["rating", "pros", "cons"]
    print(f"Dados ordenados pela data (mais recente a mais antigo). Coluna de data excluída.")
    return df

def generate_unique_ids(df):
    """Gera IDs únicos para cada linha com base nas colunas 'rating', 'pros' e 'cons'."""
    print("Gerando IDs únicos...")
    
    def generate_id(row):
        unique_string = f"{row['rating']}_{row['pros']}_{row['cons']}"
        return hashlib.sha256(unique_string.encode()).hexdigest()
    
    df["id"] = df.apply(generate_id, axis=1)
    
    if df["id"].is_unique:
        print("IDs únicos gerados com sucesso!")
    else:
        raise ValueError("Conflitos de ID detectados!")
    
    return df

def save_to_parquet(df, parquet_path):
    """Salva o DataFrame em um arquivo Parquet."""
    print(f"Salvando os dados no arquivo Parquet: {parquet_path}...")
    df.to_parquet(parquet_path, index=False)
    print("Dados salvos com sucesso!")

def main():
    """Função principal para executar o fluxo de trabalho."""
    df = load_csv(CSV_PATH)
    df = clean_data(df)
    df = generate_unique_ids(df)
    #save_to_parquet(df, PARQUET_PATH)
    print("Processo concluído!")

if __name__ == "__main__":
    main()