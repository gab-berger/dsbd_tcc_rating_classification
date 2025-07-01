import os
import json
import numpy as np
import pandas as pd
from openai import OpenAI

def load_api_key(path: str = 'data/openai_api_key.json') -> str:
    with open(path, 'r') as f:
        data = json.load(f)
    return data['api_key']

client = OpenAI(api_key=load_api_key())

def load_comments(path: str = 'data/comments_sample.parquet') -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Comments loaded! ({len(df)} total)")
    return df

def select_comments_to_predict(
    comments: pd.DataFrame,
    predictions: pd.DataFrame,
    model: str
) -> pd.DataFrame:
    if not predictions.empty:
        processed = predictions.loc[predictions['model'] == model, 'id']
        return comments[~comments['id'].isin(processed)].copy()
    return comments.copy()

def generate_system_prompt(lang: str) -> str:
    if lang == 'pt':
        return (
            "Você é um Analista Sênior de RH e Especialista em Análise de Sentimentos, "
            "avaliando feedbacks de funcionários em uma escala de 1 a 5. Seja CRÍTICO e CONSISTENTE.\n\n"
            "Critérios de Avaliação:\n"
            "[1] Experiência extremamente negativa:\n"
            "    - Não há menções positivas\n"
            "[2] Experiência de negativa a neutra:\n"
            "    - As menções negativas são a maioria\n"
            "    - NÃO PODE ser uma experiência positiva\n"
            "[3] Experiência neutra ou mista:\n"
            "    - É uma experiência equilibrada\n"
            "    - É uma experiência mista\n"
            "    - NÃO PODE SER extremamente positiva NEM extremamente negativa\n"
            "[4] Experiência de positiva a neutra:\n"
            "    - As menções positivas são a maioria\n"
            "    - NÃO PODE ser uma experiência negativa\n"
            "[5] Experiência extremamente positiva:\n"
            "    - Não há menções negativas\n\n"
            "Regras:\n"
            "- Responda apenas o JSON especificado, sem texto adicional.\n"
            "- Formato: [{\"id\":\"<id>\",\"rating\":<1-5>}, ...]"
        )
    else:
        return (
            "You are a Senior HR Analyst and Sentiment Analysis Expert, rating employee feedback on a 1-5 scale. "
            "Be CRITICAL and CONSISTENT.\n\n"
            "Rating Criteria:\n"
            "[1] Extremely negative experience:\n"
            "    - There are no positive mentions\n"
            "[2] Negative to neutral experience:\n"
            "    - The negative mentions are the majority\n"
            "    - CANNOT be a positive experience\n"
            "[3] Neutral or mixed experience:\n"
            "    - It is a balanced experience\n"
            "    - It is mixed experience\n"
            "    - CANNOT BE extremely positive NOR extremely negative\n"
            "[4] Positive to neutral experience:\n"
            "    - The positive mentions are the majority\n"
            "    - CANNOT be a negative experience\n"
            "[5] Extremely positive experience:\n"
            "    - There are no negative mentions\n\n"
            "Rules:\n"
            "- Respond only with the specified JSON, no extra text.\n"
            "- Format: [{\"id\":\"<id>\",\"rating\":<1-5>}, ...]"
        )

def generate_user_prompt(lang: str, items: list[dict]) -> str:
    payload = json.dumps(items, ensure_ascii=False)
    if lang == 'pt':
        return (
            "Classifique cada item da lista abaixo retornando apenas uma array JSON no formato especificado:\n"
            + payload
        )
    else:
        return (
            "Classify each item in the list below, returning only a JSON array in the specified format:\n"
            + payload
        )

function_schema = {
    "name": "predict_ratings",
    "description": "Retorna lista de objetos com id e rating",
    "parameters": {
        "type": "object",
        "properties": {
            "predictions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id":     {"type": "string"},
                        "rating": {"type": "integer", "minimum": 1, "maximum": 5}
                    },
                    "required": ["id", "rating"]
                }
            }
        },
        "required": ["predictions"]
    }
}

def llm_rating_predict_batch(
    comments: pd.DataFrame,
    model: str,
    batch_size: int = 20
) -> list[dict]:
    all_preds = []

    for lang, df_lang in comments.groupby('language'):
        system_msg = generate_system_prompt(lang)
        chunks = np.array_split(df_lang, max(1, len(df_lang) // batch_size + 1))

        for chunk in chunks:
            items = [
                {"id": r["id"], "pros": r["pros"], "cons": r["cons"]}
                for _, r in chunk.iterrows()
            ]
            user_msg = generate_user_prompt(lang, items)

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg}
                ],
                functions=[function_schema],
                function_call={"name": "predict_ratings"},
                temperature=0.1
            )

            args = resp.choices[0].message.function_call.arguments
            payload = json.loads(args)

            now = pd.Timestamp.now()
            for pred in payload.get('predictions', []):
                all_preds.append({
                    'id':              pred['id'],
                    'model':           model,
                    'rating':          pred['rating'],
                    'prediction_time': None,
                    'ts_prediction':   now,
                    'extra_info':      {
                        "prompt_tokens":     resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens":      resp.usage.total_tokens
                    }
                })

    return all_preds

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

def main(eligible_comments: pd.DataFrame) -> None:
    LLM_PREDICTIONS_PATH = 'data/llm_predictions.parquet'
    MODEL = "gpt-3.5-turbo-16k"

    try:
        existing_predictions = pd.read_parquet(LLM_PREDICTIONS_PATH)
    except FileNotFoundError:    
        try:
            existing_predictions = pd.read_parquet(LLM_PREDICTIONS_PATH+'.bak')
        except FileNotFoundError:
            existing_predictions = pd.DataFrame()

    comments_to_predict = select_comments_to_predict(
        eligible_comments, 
        existing_predictions, 
        MODEL
    ).sort_values(by='language')

    predictions = llm_rating_predict_batch(
        comments_to_predict,
        MODEL,
        batch_size=25
        )
    new_predictions = pd.DataFrame(predictions)
    
    try:
        existing_predictions = pd.read_parquet(LLM_PREDICTIONS_PATH)
        updated_predictions = pd.concat([existing_predictions, new_predictions], ignore_index=True)
    except FileNotFoundError:
        updated_predictions = new_predictions
    
    safe_save_to_parquet(updated_predictions, LLM_PREDICTIONS_PATH)

if __name__ == '__main__':
    eligible_comments = load_comments()
    main(eligible_comments)