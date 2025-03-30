import os
os.environ['OLLAMA_USE_GPU'] = '1'
os.environ['OLLAMA_GPU_VENDOR'] = 'AMD'
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import pandas as pd
from ollama import ChatResponse, chat
from time import time
from collections import Counter
from pydantic import BaseModel

import warnings
warnings.filterwarnings("ignore")

def load_existing_predictions(output_filename: str) -> {pd.DataFrame}:
    output_filename = output_filename
    if os.path.exists(output_filename):
        return pd.read_parquet(output_filename)
    else:
        return pd.DataFrame(columns=['id', 'rating', 'tries', 'prediction_time', 'ts_prediction'])

def load_comments() -> pd.DataFrame:
    start_time = time()
    print('Loading comments.parquet...')
    comments_df = pd.read_parquet('data/comments.parquet')
    elapsed_time = int(time() - start_time)
    print(f'comments.parquet loaded! ({elapsed_time}s)')
    return comments_df

def slice_comments(comments_df, row_start:int=None, row_end:int=None) -> pd.DataFrame:
    if row_start is None and row_end is None:
        pass
    else:
        if row_end is None:
            row_end = len(comments_df)
        comments_df = comments_df.iloc[row_start:row_end]
    return comments_df

def filter_to_predict_comments(comments_df, output_filename:str) -> pd.DataFrame:
    start_time = time()
    print(f'Loading {output_filename}...')
    pred_df = load_existing_predictions(output_filename)
    elapsed_time = round(time() - start_time, 2)
    print(f'{output_filename} loaded! ({elapsed_time}s)')

    start_time = time()
    print('Filtering out already analyzed comments...')
    analyzed_ids = set(pred_df['id'].unique())
    remaining_df = comments_df[~comments_df['id'].isin(analyzed_ids)]
    elapsed_time = round(time() - start_time, 2)
    print(f'Filtering done! ({elapsed_time}s)')

    return remaining_df, pred_df

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

def llm_query(prompt: str, model: str, temperature:int = 0.3) -> list[int,int]:
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
                    'content': prompt
                    }
                ],
            format=Prediction.model_json_schema(),
            options={
                'temperature': temperature,
                'num_predict': 10
                }
            )
        rating = Prediction.model_validate_json(response.message.content)
        return [int(rating.rating), int(response.eval_duration)]
    
    except Exception as e:
        return None

def process_comment(comment_row, model: str, prediction_repeat_target:int, temperature:float) -> dict:
    user_prompt = generate_user_prompt(comment_row['pros'], comment_row['cons'])   
    
    ratings = []
    eval_times = []
    tries = 0
    for i in range(prediction_repeat_target):
        tries += 1
        rating, eval_time = llm_query(user_prompt, model, temperature)

        ratings.append(rating)
        eval_times.append(eval_time)

        counts = Counter(ratings)
        most_common_rating, count = counts.most_common(1)[0]
        
        if count == prediction_repeat_target:
            break
    
    return {
        'id': comment_row['id'],
        'rating': most_common_rating if ratings else None,
        'repeat_target':prediction_repeat_target,
        'tries': tries,
        'temperature': temperature,
        'prediction_time': round(sum(eval_times)/1e9,2),
        'ts_prediction': pd.Timestamp.now()
    }

def save_predictions(pred_df: pd.DataFrame, output_filename: str):
    pred_df.to_parquet(output_filename, index=False)

def main(model:str, comments):
    SAVE_INTERVAL = 1
    PREDICTION_REPEAT_TARGET = 5
    LLM_TEMPERATURE = 0.3

    output_filename = f"data/pred_{model}.parquet" 
    remaining_df, pred_df = filter_to_predict_comments(comments, output_filename)

    print(f"{'='*60}\nStarting predictions with model {model}... [loop:{len(remaining_df)}][total:{len(pred_df)}]\n{'='*60}")
    
    new_predictions = []
    count = 0
    for idx, row in remaining_df.iterrows():
        prediction = process_comment(row, model, PREDICTION_REPEAT_TARGET, LLM_TEMPERATURE)
        new_predictions.append(prediction)
        print(f"[{idx+1}/{len(remaining_df)}] {row['id'][:4]}...{row['id'][-5:]} done! Prediction: {prediction['rating']} ({int(prediction['prediction_time'])}s/{int(prediction['tries'])}t)")
        count += 1
        
        if count % SAVE_INTERVAL == 0:
            temp_df = pd.DataFrame(new_predictions)
            pred_df = pd.concat([pred_df, temp_df], ignore_index=True)
            save_predictions(pred_df, output_filename)
            new_predictions = []
    
    if new_predictions:
        temp_df = pd.DataFrame(new_predictions)
        pred_df = pd.concat([pred_df, temp_df], ignore_index=True)
        save_predictions(pred_df, output_filename)
    
    print(f"{'='*60}\nModel {model} finished predictions! [loop:{len(remaining_df)}][total:{len(pred_df)}]\n{'='*60}")

if __name__ == '__main__':
    LOOP_RANGE = 50
    models  = [
        'deepseek-r1:1.5b',
        'stablelm2',
        'llama3.1',
        'llama3.2',
        'deepseek-r1:8b',
        'llama2:7b',
        'llama2:13b',
        'stablelm2:12b'
    ][0:1]

    all_comments = load_comments()
    for n in range(0, 15000, LOOP_RANGE):
        to_predict_comments = slice_comments(all_comments, n, n+LOOP_RANGE)
        for model in models:
            main(model, to_predict_comments)
    
    print('All work done! :)')