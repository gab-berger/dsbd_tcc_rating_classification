import pandas as pd
import sqlite3
import subprocess
from time import time

DB_PATH = "comments.db"
COMS_TABLE_NAME = "comments"
CLASS_TABLE_NAME = ""

def get_row_info(conn, table_name, row_n):
    row_id_query = f"SELECT id FROM {table_name} LIMIT 1 OFFSET {row_n}"
    row_rating_query = f"SELECT rating FROM {table_name} LIMIT 1 OFFSET {row_n}"
    row_pros_query = f"SELECT pros FROM {table_name} LIMIT 1 OFFSET {row_n}"
    row_cons_query = f"SELECT cons FROM {table_name} LIMIT 1 OFFSET {row_n}"
    
    row_id = pd.read_sql(row_id_query, conn).iloc[0, 0]
    row_rating = int(pd.read_sql(row_rating_query, conn).iloc[0, 0])
    row_pros = pd.read_sql(row_pros_query, conn).iloc[0, 0] or ""
    row_cons = pd.read_sql(row_cons_query, conn).iloc[0, 0] or "" 

    return row_id, row_rating, row_pros, row_cons

def generate_prompt_flag(rating, pros, cons):
    return f"""Analyze the following feedback and determine if the employee's rating is consistent with their pros and cons:

    Rating: {rating} (Range: 1-5)
    Pros: {pros}
    Cons: {cons}

    Instructions:
    - The rating ranges from 1 (lowest) to 5 (highest).
    - Return '1' if the rating is clearly NOT consistent with the pros and cons.
    - Return '0' if the rating IS consistent with the pros and cons or if there is any uncertainty.

    Output rules:
    - ONLY respond with a single character: either '0' or '1'.
    - Do NOT return any explanations, justifications, additional text, or code.
    - Responses that include anything other than '0' or '1' are INVALID.
    - The maximum length of your response is ONE character.
    - STRICTLY FOLLOW THESE RULES."""

def generate_prompt_rating(pros, cons):
    return f"""Evaluate the following employee feedback and determine the overall rating based on the provided pros and cons:

    Pros: {pros}
    Cons: {cons}

    Rating scale (1 to 5):
        - 1: Very poor experience
        - 2: Below average experience
        - 3: Neutral or mixed experience
        - 4: Good experience
        - 5: Excellent experience

    Output requirements:
        - Respond with ONLY a single character: '1', '2', '3', '4', or '5'.
        - Do NOT include any explanations, justifications, extra text, or code.
        - Responses containing anything other than '1', '2', '3', '4', or '5' are INVALID.
        - Your response must be exactly ONE character long.
        - STRICTLY follow to these rules.
    """

def llm_query(prompt:str, model:str='llama3.2'):
    command = ['ollama', 'run', model]
    start_time = time()

    try:
        result = subprocess.run(command, input=prompt, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error executing Ollama:\n{result.stderr}")
            return None
        
        output = result.stdout.strip()
        if output[-1] in ['1', '2', '3', '4', '5']:
            print(f"LLM Response: {output[-1]}")
            return int(output[-1])
        else:
            print(f"Unexpected response: {output}")
            return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def calculate_inconsistency_percentage(prompt, model, iterations=10):
    inconsistent_count = 0
    valid_outputs = 0
    while valid_outputs < iterations:
        result = llm_query(prompt, model)
        if result in [0, 1]:
            inconsistent_count += result
            valid_outputs += 1

    inconsistency_percentage = (inconsistent_count / valid_outputs) * 100
    print(f"Inconsistencies detected: {inconsistent_count}/{valid_outputs} ({inconsistency_percentage:.2f}%)")
    return inconsistency_percentage

def calculate_prediction_average(prompt, model, iterations=10):
    avg_sum = 0
    valid_outputs = 0
    while valid_outputs < iterations:
        result = llm_query(prompt, model)
        if result in [1, 2, 3, 4, 5]:
            avg_sum += result
            valid_outputs += 1

    prediction_avg = (avg_sum / valid_outputs)
    print(f"Average rating: {avg_sum}/{valid_outputs} ({prediction_avg:.2f})")
    return prediction_avg

def main(model, iterations):
    """Main function to execute the workflow."""
    conn = sqlite3.connect(DB_PATH)
    id, rating, pros, cons = get_row_info(conn, COMS_TABLE_NAME, 10)
    conn.close()

    prompt = generate_prompt_flag(rating, pros, cons)
    print('\n',"="*80,'\n',prompt,'\n',"="*80,'\n')
    calculate_inconsistency_percentage(prompt, model, iterations=iterations)

    # prompt = generate_prompt_rating(pros, cons)
    # print('\n',"="*80,'\n',prompt,'\n',"="*80,'\n')
    # calculate_prediction_average(prompt, model, iterations=iterations)

    # llm_query(prompt, model='llama3.1')

if __name__ == "__main__":
    model = [
        # 'stablelm2',
        # 'llama3.2',
        # 'llama3.1',
        'stablelm2:12b'
    ]
    
    main(model[0], 20)