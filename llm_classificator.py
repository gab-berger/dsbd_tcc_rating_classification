import os
import pandas as pd
from ollama import ChatResponse, chat
from time import time
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

def load_existing_predictions(output_filename: str) -> {pd.DataFrame}:
    output_filename = 'data/'+output_filename
    if os.path.exists(output_filename):
        return pd.read_parquet(output_filename)
    else:
        return pd.DataFrame(columns=['id', 'classification', 'num_tries', 'prediction_time', 'ts_prediction'])

def load_data(output_filename:str) -> pd.DataFrame:
    start_time = time()
    print('Loading comments.parquet...')
    comments_df = pd.read_parquet('data/comments.parquet')
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

def process_comment(comment_row, model: str, num_tries: int) -> dict:
    prompt = generate_prompt_rating(comment_row['pros'], comment_row['cons'])
    ratings = []
    last_rating = 0
    start_time = time()
    
    for i in range(num_tries):
        rating = llm_query(prompt, model)

        if rating == last_rating:
            break
        last_rating = rating
        
        if rating is not None:
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
        'num_tries': len(ratings),
        'prediction_time': elapsed_time,
        'ts_prediction': ts_prediction
    }

def save_predictions(pred_df: pd.DataFrame, output_filename: str):
    start_time = time()
    pred_df.to_parquet(output_filename, index=False)
    elapsed_time = round(time() - start_time, 2)
    print(f"Predictions saved to {output_filename} ({elapsed_time}s)")

def main(model: str, num_tries: int = 5, save_interval: int = 10):
    output_filename = f"pred_{model}.parquet" 
    remaining_df, pred_df = load_data(output_filename)

    print(f"Starting predictions with model {model}...")
    new_predictions = []
    count = 0
    
    for idx, row in remaining_df.iterrows():
        prediction = process_comment(row, model, num_tries)
        new_predictions.append(prediction)
        print(f"[{idx+1}/{save_interval}] Comment {row['id'][:4]}...{row['id'][-5:]} done! Prediction: {prediction['classification']} ({int(prediction['prediction_time'])}s)")
        count += 1
        
        if count % save_interval == 0:
            temp_df = pd.DataFrame(new_predictions)
            pred_df = pd.concat([pred_df, temp_df], ignore_index=True)
            save_predictions(pred_df, output_filename)
            new_predictions = []
    
    if new_predictions:
        temp_df = pd.DataFrame(new_predictions)
        pred_df = pd.concat([pred_df, temp_df], ignore_index=True)
        save_predictions(pred_df, output_filename)
    
    print(f"Total new comments processed: {count}")

if __name__ == '__main__':
    model_name  = [
        'deepseek-r1:1.5b',
        'stablelm2',
        'llama3.1',
        'llama3.2',
        'deepseek-r1:8b',
        'stablelm2:12b',
        'llama2:7b',
        'llama2:13b',
        'vicuna',
        'falcon'
    ][3]
    main(model_name, num_tries=5, save_interval=5)