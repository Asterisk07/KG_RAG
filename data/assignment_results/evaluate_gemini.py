import pandas as pd
import json


import argparse

parser = argparse.ArgumentParser(description='Process some modes.')
parser.add_argument('--mode', type=int, default=0,
                    help='Mode of operation (default: 1)')

args = parser.parse_args()
MODE = (args.mode)


# Define file paths for the CSV files
file_path1 = f'data/assignment_results/gemini_1.5_flash_kg_rag_based_mcq_{MODE}.csv'


# Load the CSV files into DataFrames
df1 = pd.read_csv(file_path1)

# Define a function to check if the correct answer is present in the LLM answer


def contains_correct_answer(row):
    try:
        return row['correct_answer'] == json.loads(row['llm_answer'].replace('```', '').replace('\n', '').replace('json', '').replace('{{', '{').replace('}}', '}').split('}')[0] + '}')['answer']
    except:
        return False


# Apply the function to each row of the DataFrames
df1['is_correct'] = df1.apply(contains_correct_answer, axis=1)

# Calculate the percentage of correct answers
correct_rate1 = df1['is_correct'].mean() * 100
print(f"Correct Answer Rate for {file_path1}: {correct_rate1:.2f}%")
