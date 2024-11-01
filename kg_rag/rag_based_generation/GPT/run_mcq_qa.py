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
parser.add_argument("--trial", action='store_true',
                    help="Generate additional df or not")
args = parser.parse_args()

# Access the arguments
CHAT_MODEL_ID = args.chat_model_id
MODE = str(args.mode)
EVALUATE_FLAG = args.eval
INTERACTIVE_FLAG = args.interactive
TRIAL_FLAG = args.trial
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


def _final_answer(row):
    try:
        return json.loads(row['llm_answer'].replace('```', '').replace('\n', '').replace('json', '').replace('{{', '{').replace('}}', '}').split('}')[0] + '}')['answer']
    except:
        return False


def _append_prior():
    prior = '\nProvenance & Symptoms information is useless. \nSimilar diseases tend to have similar gene associations.'
    return prior


def jsonize_prompt(x, variant_flag=True):
    gene_key = 'Genetic Associations'
    variant_key = 'Variant Associations'
    sentences = x.strip().split('.')
    d1 = dict()

    for sentence in sentences:
        if (len(sentence)) == 0:
            continue
        sentence_list = sentence.split('and Provenance of this association is')

        reason = None
        if len(sentence_list) == 2:
            sentence, reason = sentence_list
            reason = reason.strip()
        else:
            sentence = sentence_list[0]
        # print(sentence, sentence.split('associates'),sep  = '\n', end = '--------\n\n')
        u, v = sentence.split('associates')
        u = u.strip().split(' ', maxsplit=1)
        v = v.strip().split(' ', maxsplit=1)

        u[0] = u[0].strip()
        u[1] = u[1].strip()

        v[0] = v[0].strip()
        v[1] = v[1].strip()

        if v[0] == 'Disease':
            u, v = v, u

        if (u[0] == 'Disease'):
            disease = u[0]
            if (v[0] == 'Variant') and variant_flag:
                d3 = dict()
                d3['Variant'] = v[1]

                if reason:
                    d3['Provenance'] = reason

                if disease not in d1:
                    d1[disease] = dict()

                if variant_key not in d1[disease]:
                    d1[disease][variant_key] = list()

                d1[disease][variant_key].append(d3)

            elif (v[0] == 'Gene'):
                d3 = dict()
                d3['Gene'] = v[1]

                if reason:
                    d3['Provenance'] = reason

                if disease not in d1:
                    d1[disease] = dict()

                if gene_key not in d1[disease]:
                    d1[disease][gene_key] = list()

                d1[disease][gene_key].append(d3)

                pass

    json_data = json.dumps(d1)
    return json_data


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

    if TRIAL_FLAG:
        context_list = []
        entity_list = []
        prompt_list = []

    for index, row in tqdm(question_df.iterrows(), total=len(question_df)):
        try:
            question = row["text"]
            # print("fetching context")
            context, entity = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME,
                                               QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID, return_entities_flag=True)
            # print(f"fetched context : {context[:10]}")

            if MODE == "0":
                ### MODE 0: Original KG_RAG                     ###
                pass

            elif MODE == "1":

                ### MODE 1: jsonlize the context from KG search ###
                ### Please implement the first strategy here    ###

                output = '...'

            elif MODE == "2":
                ### MODE 2: Add the prior domain knowledge      ###
                ### Please implement the second strategy here   ###
                # output = '...'
                context += _append_prior()

            elif MODE == "3":
                ### MODE 3: Combine MODE 1 & 2                  ###
                ### Please implement the third strategy here    ###
                output = '...'

            elif MODE == '11':

                context = jsonize_prompt(context, variant_flag=False)

            elif MODE == '12':

                context = jsonize_prompt(context, variant_flag=True)

            elif MODE == '21':

                context = jsonize_prompt(context, variant_flag=False)
                context += _append_prior()

            elif MODE == '22':

                context = jsonize_prompt(context, variant_flag=True)
                context += _append_prior()

            enriched_prompt = "Context: " + context + "\n" + "Question: " + question

            output = get_Gemini_response(
                enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            answer_list.append((row["text"], row["correct_node"], output))

            if TRIAL_FLAG:
                context_list.append(context)
                prompt_list.append(enriched_prompt)
                entity_list.append(entity)

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

    if TRIAL_FLAG:
        # print(f'lenghts are {len(entity_list)}, {len(context_list)}')
        answer_df['entity'] = entity_list
        answer_df['context'] = context_list
        answer_df['prompt'] = prompt_list
        answer_df['final_ans'] = answer_df.apply(_final_answer, axis=1)
        answer_df.to_csv(f'try_data_{MODE}.csv', index=False)


if __name__ == "__main__":
    main()

if EVALUATE_FLAG:
    command = f"python data/assignment_results/evaluate_gemini.py --mode {MODE}"
    exit_code = os.system(command)

# python -m kg_rag.rag_based_generation.GPT.run_mcq_qa gemini-1.5-flash --mode 0 --eval --trial ;
'''
run in git bash
python -m kg_rag.rag_based_generation.GPT.run_mcq_qa gemini-1.5-flash --mode 2 --eval --trial ;
python -m kg_rag.rag_based_generation.GPT.run_mcq_qa gemini-1.5-flash --mode 11 --eval --trial ;
python -m kg_rag.rag_based_generation.GPT.run_mcq_qa gemini-1.5-flash --mode 12 --eval --trial ;
python -m kg_rag.rag_based_generation.GPT.run_mcq_qa gemini-1.5-flash --mode 21 --eval --trial ;
python -m kg_rag.rag_based_generation.GPT.run_mcq_qa gemini-1.5-flash --mode 22 --eval --trial ;
'''
