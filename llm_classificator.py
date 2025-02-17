import os
import pandas as pd
from ollama import ChatResponse, chat
from time import time
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

def load_existing_predictions(output_filename: str) -> {pd.DataFrame}:
    output_filename = output_filename
    if os.path.exists(output_filename):
        return pd.read_parquet(output_filename)
    else:
        return pd.DataFrame(columns=['id', 'classification', 'num_tries', 'prediction_time', 'ts_prediction'])

def load_data(output_filename:str, row_start:int=None, row_end:int=None) -> pd.DataFrame:
    start_time = time()
    print('Loading comments.parquet...')
    comments_df = pd.read_parquet('data/comments.parquet')
    
    if row_start is None and row_end is None:
        pass
    else:
        if row_end is None:
            row_end = len(comments_df)
        comments_df = comments_df.iloc[row_start:row_end]
    
    elapsed_time = int(time() - start_time)
    print(f'comments.parquet loaded! ({elapsed_time}s)')

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

def generate_prompt_rating(pros:str, cons:str) -> str:
    return f"""You are a Sentiment Analysis Specialist, and your goal is to evaluate the following employee feedback and determine the overall rating based on the provided pros and cons:

Pros: {pros}
Cons: {cons}

Rating scale (1 to 5):
    - 1: Extremely negative experience
    - 2: Negative experience
    - 3: Neutral or mixed experience
    - 4: Positive experience
    - 5: Extremely positive experience

Output requirements:
    - Respond with ONLY a single character: '1', '2', '3', '4', or '5'.
    - Do NOT include any explanations, justifications, extra text, or code.
    - Your response must be exactly ONE character long.
"""

def llm_query(prompt: str, model: str) -> int:
    try:
        response: ChatResponse = chat(
            model=model,
            messages=[{'role': 'user','content': prompt}]
            )
        answer = str(response['message']['content']).strip()

        for char in reversed(answer):
            if char in ['1', '2', '3', '4', '5']:
                return int(char)
        return None
    
    except Exception as e:
        return None

def process_comment(comment_row, model: str, num_tries:int) -> dict:
    prompt = generate_prompt_rating(comment_row['pros'], comment_row['cons'])
    ratings = []
    last_rating = 42
    tries = 0
    start_time = time()
    
    for i in range(num_tries):
        tries += 1
        rating = llm_query(prompt, model)

        if rating == last_rating:
            break
        last_rating = rating

        ratings.append(rating)
        counts = Counter(ratings)
        most_common_rating, count = counts.most_common(1)[0]
        
        if count > num_tries // 2:
            break
    
    elapsed_time = round(time() - start_time, 2)
    ts_prediction = pd.Timestamp.now()
    
    return {
        'id': comment_row['id'],
        'classification': most_common_rating if ratings else None,
        'tries': tries,
        'prediction_time': elapsed_time,
        'ts_prediction': ts_prediction
    }

def save_predictions(pred_df: pd.DataFrame, output_filename: str):
    start_time = time()
    pred_df.to_parquet(output_filename, index=False)
    elapsed_time = round(time() - start_time, 2)
    print(f"Predictions saved to {output_filename} ({elapsed_time}s)")

def main(model:str, df_interval:list=[None,None]):
    SAVE_INTERVAL = 5
    TOTAL_LLM_TRIES = 5

    row_start, row_end = df_interval

    output_filename = f"data/pred_{model}.parquet" 
    remaining_df, pred_df = load_data(output_filename, row_start, row_end)

    print(f"{'='*60}\nStarting predictions with model {model}... [{row_start} -> {row_end}]\n{'='*60}")
    new_predictions = []
    count = 0
    
    for idx, row in remaining_df.iterrows():
        prediction = process_comment(row, model, TOTAL_LLM_TRIES)
        new_predictions.append(prediction)
        print(f"[{idx+1}/{int(row_end-row_start+1)}] {row['id'][:4]}...{row['id'][-5:]} done! Prediction: {prediction['classification']} ({int(prediction['prediction_time'])}s/{int(prediction['tries'])}t)")
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
    
    print(f"{'='*60}\nModel {model} finished predictions! [{row_start} -> {row_end}]\n{'='*60}")

if __name__ == '__main__':
    models  = [
        'deepseek-r1:1.5b',
        'stablelm2',
        'llama3.1',
        'llama3.2',
        'deepseek-r1:8b',
        'llama2:7b',
        'llama2:13b',
        'stablelm2:12b',
        #'vicuna',
        #'falcon'
    ]
    for n in range(0, 100, 1000):
        for model in models:
            main(model, [n, n+99])
    
    print('All work done! :)')