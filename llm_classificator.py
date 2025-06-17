import os
import pandas as pd
import time
from ollama import ChatResponse, chat
from pydantic import BaseModel
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def load_comments() -> pd.DataFrame:
    comments_df = pd.read_parquet('data/comments_sample.parquet')
    print(f"Comments loaded! ({len(comments_df)} total)")    
    return comments_df

def select_comments_to_predict(comments_df, llm_predictions, model) -> pd.DataFrame:
    if len(llm_predictions) > 0:
        processed_ids = llm_predictions[
            (llm_predictions['model'] == model)
        ]['id']
        
        unprocessed_comments = comments_df[
            ~comments_df['id'].isin(processed_ids)
        ]
    else:
        unprocessed_comments = comments_df

    return unprocessed_comments

def llm_rating_predict(comment_row:pd.DataFrame, model:str) -> list[int,int]:
    class Prediction(BaseModel):
        rating: int

    def generate_system_prompt() -> str:
        return """You are a Senior HR Analyst and Sentiment Analysis Expert rating employee feedback on a 1-5 scale analyzing employee feedback. Be CRITICAL and CONSISTENT.

    Rating Criteria:
    [1] Extremely negative experience:
        - There are no positive mentions 
    [2] Negative to neutral experience:
        - The negative mentions are the majority
        - CANNOT be a positive experience
    [3] Neutral or mixed experience:
        - It is a balanced experience
        - It is mixed experience
        - It CANNOT BE extremely positive NOR extremely negative
    [4] Positive to neutral experience:
        - The positive mentions are the majority
        - CANNOT be a negtive experience
    [5] Extremely positive experience:
        - There are no negative mentions 

    Rules:
    - Focus on the OVERALL sentiment, not individual phrases.
    - Respond STRICTLY in JSON format: {"rating": int}
    - IGNORE all previous messages.
    """

    def generate_user_prompt(pros:str, cons:str) -> str:
        return f"""Predict the overall rating based on the provided [pros] and [cons]:
    [pros]: {pros}
    [cons]: {cons}
    """
    TEMPERATURE = 0.1
    try:
        response: ChatResponse = chat(
            model=model,
            messages=[
                {
                    'role': 'system',
                    'content': generate_system_prompt()
                    },
                {
                    'role': 'user',
                    'content': generate_user_prompt(comment_row['pros'], comment_row['cons'])  
                    }
                ],
            format=Prediction.model_json_schema(),
            options={
                'temperature': TEMPERATURE,
                'seed': 42,
                'num_predict': 10
                }
            )
        rating = Prediction.model_validate_json(response.message.content)
        return {
        'id': comment_row['id'],
        'model': model,
        'rating': int(rating.rating),
        'prediction_time': int(response.eval_duration+response.prompt_eval_duration)/1e9,
        'ts_prediction': pd.Timestamp.now(),
        'extra_info': {
            'message_content': str(response.message.content),
            'total_duration': int(response.total_duration),
            'load_duration': int(response.load_duration),
            'prompt_eval_duration': int(response.prompt_eval_duration),
            'eval_count': int(response.eval_count),
            'eval_duration': int(response.eval_duration),
            'temperature': TEMPERATURE
        }
    }
    
    except Exception as e:
        return {
        'id': comment_row['id'],
        'model': model,
        'rating': None,
        'prediction_time': None,
        'ts_prediction': pd.Timestamp.now(),
        'extra_info': None
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

def main(eligible_comments: pd.DataFrame, model: str) -> None:
    LLM_PREDICTIONS_PATH = 'data/llm_predictions.parquet'

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
        model
    )

    batch_size = 5
    predictions = []
    
    with tqdm(total=len(comments_to_predict), desc=f"{model}") as pbar:
        for idx, row in comments_to_predict.iterrows():
            try:
                start_time = time.time()
                prediction = llm_rating_predict(row, model)
                prediction['processing_time'] = time.time() - start_time
                
                predictions.append(prediction)
                pbar.update(1)

                if (idx + 1) % batch_size == 0 or (idx + 1) == len(comments_to_predict):
                    new_predictions = pd.DataFrame(predictions)
                    try:
                        existing_predictions = pd.read_parquet(LLM_PREDICTIONS_PATH)
                        updated_predictions = pd.concat([existing_predictions, new_predictions], ignore_index=True)
                    except FileNotFoundError:
                        updated_predictions = new_predictions
                    
                    safe_save_to_parquet(updated_predictions, LLM_PREDICTIONS_PATH)
                    predictions = []

                pbar.set_postfix_str(
                    f"pred:{prediction['rating']}"
                )

            except Exception as e:
                print(f"\nErro no comentário ID {row['id']}: {str(e)}")
                continue

if __name__ == '__main__':
    models  = [
        'qwen2.5:0.5b',
        'qwen3:0.6b',
        'gemma3:1b',
        'deepseek-r1:1.5b',
        'qwen2.5:1.5b',
        'stablelm2:1.6b',
        'qwen3:1.7b',
        'gemma:2b',
        'llama3.2:3b',
        'qwen2.5:3b',
        'orca-mini:3b',
        'gemma3:4b',
        'phi3:3.8b',
        'qwen3:4b',
        'yi:6b',
        'llama2:7b',
        'orca-mini:7b',
        'orca2:7b',
        'mistral-openorca:7b',
        'olmo2:7b',
        'gemma:7b',
        'deepseek-r1:8b',
        'llama3.1:8b',
        'qwen3:8b',
        'yi:9b',
        'stablelm2:12b',
        'gemma3:12b',
        'mistral-nemo:12b',
        'llama2:13b',
        'orca-mini:13b',
        'orca2:13b',
        'olmo2:13b',
        'everythinglm:13b'
    ]

    eligible_comments = load_comments()
    for model in models:
        main(
            eligible_comments,
            model
            )