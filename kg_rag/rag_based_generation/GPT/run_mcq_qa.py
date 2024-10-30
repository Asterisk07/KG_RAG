'''
This script takes the MCQ style questions from the csv file and save the result as another csv file.
Before running this script, make sure to configure the filepaths in config.yaml file.
Command line argument should be either 'gpt-4' or 'gpt-35-turbo'
'''

from kg_rag.utility import *
from tqdm import tqdm

import os

# OLD PARSING
# import sys
# CHAT_MODEL_ID = sys.argv[1]

# NEW Parsing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('chat_model_id', type=str, help='The ID of the chat model')
parser.add_argument('--mode', type=int, default=0, help='Mode (default: 0)')
parser.add_argument("--num", type=int, default=-1,
                    help="Number of questions to run (default: 10).")
parser.add_argument("--eval", action='store_true', help="Evaluate or not")
parser.add_argument("--interactive", action='store_true',
                    help="Interactive or not")
args = parser.parse_args()

# Access the arguments
CHAT_MODEL_ID = args.chat_model_id
MODE = str(args.mode)
EVALUATE_FLAG = args.eval
INTERACTIVE_FLAG = args.interactive
NUM = args.num

if NUM <= 0 and NUM != -1:
    raise Exception(
        "--num parameter must be positive, or -1 to run the whole dataframe")

print(f"Chat Model ID: {CHAT_MODEL_ID}")
print(f"Mode: {MODE}")
# print(f"cwd: {os.getcwd()}")


# raise ZeroDivisionError


QUESTION_PATH = config_data["MCQ_PATH"]
SYSTEM_PROMPT = system_prompts["MCQ_QUESTION"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(
    config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(
    config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data[
    "SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data[
    "SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]
SAVE_PATH = config_data["SAVE_RESULTS_PATH"]


CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID

save_name = "_".join(CHAT_MODEL_ID.split("-"))+"_kg_rag_based_mcq_{mode}.csv"


vectorstore = load_chroma(
    VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(
    SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)
edge_evidence = False


# MODE = "0"
### MODE 0: Original KG_RAG                     ###
### MODE 1: jsonlize the context from KG search ###
### MODE 2: Add the prior domain knowledge      ###
### MODE 3: Combine MODE 1 & 2                  ###

# INTERACTIVE_FLAG
while INTERACTIVE_FLAG:
    # print("Enter prompt:\n")
    question = input("Enter prompt (type exit to exit):\n")

    if question == 'exit':
        sys.exit(0)

    context = retrieve_context(question, vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME,
                               QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)
    # print("fetched context")
    enriched_prompt = "Context: " + context + "\n" + "Question: " + question
    # print("extracting response now")
    output = get_Gemini_response(
        enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)
    output = json.loads(output.replace('```', '').replace('\n', '').replace(
        'json', '').replace('{{', '{').replace('}}', '}').split('}')[0] + '}')['answer']
    print(f"Answer {output}")
    print("------------------------------------------")


def main():
    start_time = time.time()

    question_df = pd.read_csv(QUESTION_PATH)
    # print(question_df.shape)
    if NUM == -1:
        pass

    else:
        question_df = question_df[:NUM]
    # print(question_df.shape)

    answer_list = []
    # print(f"Running in mode {MODE}")

    for index, row in tqdm(question_df.iterrows(), total=len(question_df)):
        try:
            question = row["text"]
            if MODE == "0":
                ### MODE 0: Original KG_RAG                     ###
                # print("fetching context")
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME,
                                           QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)
                # print("fetched context")
                enriched_prompt = "Context: " + context + "\n" + "Question: " + question
                # print("extracting response now")
                output = get_Gemini_response(
                    enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            if MODE == "1":
                ### MODE 1: jsonlize the context from KG search ###
                ### Please implement the first strategy here    ###
                output = '...'

            if MODE == "2":
                ### MODE 2: Add the prior domain knowledge      ###
                ### Please implement the second strategy here   ###
                output = '...'

            if MODE == "3":
                ### MODE 3: Combine MODE 1 & 2                  ###
                ### Please implement the third strategy here    ###
                output = '...'

            answer_list.append((row["text"], row["correct_node"], output))
        except Exception as e:
            print("Error in processing question: ", row["text"])
            print("Error: ", e)
            answer_list.append((row["text"], row["correct_node"], "Error"))

    answer_df = pd.DataFrame(answer_list, columns=[
        "question", "correct_answer", "llm_answer"])
    output_file = os.path.join(SAVE_PATH, f"{save_name}".format(mode=MODE),)
    answer_df.to_csv(output_file, index=False, header=True)
    print("Save the model outputs in ", output_file)
    print("Completed in {} min".format((time.time()-start_time)/60))


if __name__ == "__main__":
    main()

if EVALUATE_FLAG:
    command = "python data/assignment_results/evaluate_gemini.py"
    exit_code = os.system(command)
