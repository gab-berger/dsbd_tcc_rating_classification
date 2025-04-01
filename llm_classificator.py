import os
import pandas as pd
import time
from ollama import ChatResponse, chat
from collections import Counter
from pydantic import BaseModel
from typing import Dict, Any
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def select_eligible_comments() -> pd.DataFrame:
    comments_df = pd.read_parquet('data/comments.parquet')
    print(f"Comments loaded! ({len(comments_df)} total)")
    manual_predictions = pd.read_parquet('data/manual_predictions.parquet')
    print(f"Manual predictions loaded! ({len(manual_predictions)} total)")
    
    return comments_df[comments_df['id'].isin(manual_predictions['id'])]

def select_comments_to_predict(comments_df, llm_predictions, model, temperature) -> pd.DataFrame:
    if len(llm_predictions) > 0:
        processed_ids = llm_predictions[
            (llm_predictions['model'] == model) & 
            (llm_predictions['temperature'] == temperature)
        ]['id']
        
        unprocessed_comments = comments_df[
            ~comments_df['id'].isin(processed_ids)
        ]
    else:
        unprocessed_comments = comments_df

    return unprocessed_comments

def llm_query(comment_row:pd.DataFrame, model:str, temperature:int = 0.3) -> list[int,int]:
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
                'temperature': temperature,
                'seed': 42,
                'num_predict': 10
                }
            )
        rating = Prediction.model_validate_json(response.message.content)
        return [int(rating.rating), int(response.eval_duration+response.prompt_eval_duration), response]
    
    except Exception as e:
        return None

def predict_rating(comment_row, model: str, temperature:float) -> dict:
    prediction_repeat_target = max(int(temperature*10),1)
    
    ratings = []
    prediction_times = []
    responses = []
    count = 0
    tries = 0
    while count < prediction_repeat_target:
        tries += 1
        rating, prediction_time, response = llm_query(comment_row, model, temperature)

        ratings.append(rating)
        prediction_times.append(prediction_time)
        responses.append(response)

        counts = Counter(ratings)
        most_common_rating, count = counts.most_common(1)[0]
        
        if count == prediction_repeat_target:
            break
    
    return {
        'id': comment_row['id'],
        'model': model,
        'temperature': temperature,
        'rating': most_common_rating if ratings else None,
        'all_predictions': ratings,
        'tries': tries,
        'repeat_target': prediction_repeat_target,
        'prediction_time': sum(prediction_times)/1e9,
        'ts_prediction': pd.Timestamp.now(),
        'llm_outputs': responses
    }

def main(eligible_comments_df: pd.DataFrame, model: str, temperature: float) -> None:
    LLM_PREDICTIONS_PATH = 'data/llm_predictions.parquet'

    try:
        existing_predictions = pd.read_parquet(LLM_PREDICTIONS_PATH)
    except FileNotFoundError:
        existing_predictions = pd.DataFrame()

    comments_to_predict = select_comments_to_predict(
        eligible_comments_df, 
        existing_predictions, 
        model, 
        temperature
    )

    batch_size = 5
    predictions = []
    
    with tqdm(total=len(comments_to_predict), desc=f"{model}_t{temperature}") as pbar:
        for idx, row in comments_to_predict.iterrows():
            try:
                start_time = time.time()
                
                prediction = predict_rating(row, model, temperature)
                prediction['processing_time'] = time.time() - start_time
                
                predictions.append(prediction)
                pbar.update(1)

                if (idx + 1) % batch_size == 0 or (idx + 1) == len(comments_to_predict):
                    new_predictions_df = pd.DataFrame(predictions)
                    try:
                        existing_predictions = pd.read_parquet(LLM_PREDICTIONS_PATH)
                        updated_predictions = pd.concat([existing_predictions, new_predictions_df], ignore_index=True)
                        updated_predictions.to_parquet(LLM_PREDICTIONS_PATH)
                    except FileNotFoundError:
                        updated_predictions = new_predictions_df
                    predictions = []

                pbar.set_postfix_str(
                    f"tries:{prediction['tries']}|"
                    f"time:{prediction['prediction_time']:.1f}s"
                )

            except Exception as e:
                print(f"\nErro no comentário ID {row['id']}: {str(e)}")
                continue

if __name__ == '__main__':
    models  = [
        'deepseek-r1:1.5b',
        'llama2:13b',
        'llama3.2',
        'llama3.1',
        'stablelm2:12b',
        'llama2:7b',
        'stablelm2',
        'deepseek-r1:8b'
    ]

    temperatures = [
        0.1,
        0.3,
        0.5,
        0.8
    ]

    eligible_comments_df = select_eligible_comments()
    
    for model in models:
        for temperature in temperatures:
            main(eligible_comments_df.iloc[0:250], model, temperature)