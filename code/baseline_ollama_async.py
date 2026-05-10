import os
import os
import re
import sys
import math
import asyncio
import warnings
import traceback
import pandas as pd
from ollama import AsyncClient
from tqdm.asyncio import tqdm_asyncio


warnings.filterwarnings("ignore")

# ==============================
# Prompts
# ==============================

prompt_execution = """
Data:
{raw_data}

Instruction:
{perturbed_prompt}

"""

# prompt_task = """
# Give the output as the markdown format like this:
# ```Output:```
# """

prompt_task = """
Give only the output without any explanation.
"""

# ==============================
# Helpers
# ==============================

def process_prompt(prompt, word_count_type, word_count):

    prompt_final = (
        prompt
        .replace("{word_count_type}", str(word_count_type))
        .replace("{word_count}", str(word_count))
    )
    
    return prompt_final + prompt_task

def extract_output(text):
    if not text:   
        return ""
    
    # match = re.search(r"```Output:\s*([\s\S]*?)```", text)
    # return match.group(1).strip() if match else ""

    return str(text).strip()

def calculate_LD(l_output, l_constraint):
    return (l_output-l_constraint)/l_constraint

def calculate_LS(LD, k1=5, k2=2):
    return 100 * math.exp(k1 * LD if LD < 0 else -k2 * LD)

# ==============================
# Async LLM Call
# ==============================

async def get_response_async(client, model_name, prompt):

    try:
        response = await client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_ctx": 65536},
        )
        return response["message"]["content"]
            
    except Exception as e:
        print("Error during LLM call:", e)
        traceback.print_exc()
        return None

# ==============================
# Row Processor
# ==============================

async def process_row(semaphore, client, df, index, word_count_type, word_count, model_name):
    async with semaphore:
        row = df.loc[index]
        prompt = process_prompt(row["task"], word_count_type, word_count)

        # print(word_count)


        limit = 3
        for attempt in range(limit):
            response = await get_response_async(
                client, model_name, prompt
            )

            df.at[index, "response"] = response
            df.loc[[index], ["output"]] = (
                df.loc[[index], "response"].apply(extract_output)
            )
            val = df.at[index, "output"]
            if pd.isna(val) or val == "":
                print(f"Attempt {attempt+1}/{limit} failed for index {index}. Retrying...")
                await asyncio.sleep(2)  # wait before retrying

        output = df.at[index, "output"]
        l_output = len(str(output).split()) if output else 0
        df.at[index, "word_count"] = l_output
        ld = calculate_LD(l_output, int(word_count))
        df.at[index, "LD"] = ld
        df.at[index, "LS"] = calculate_LS(ld)


# ==============================
# Main Async Pipeline
# ==============================

async def process_csv_async(
    input_file,
    output_file,
    model_name,
    word_count_type,
    word_count,
    max_concurrency=4,
):
    df = pd.read_csv(input_file)

    #df = df.iloc[:2]

    df["response"] = None
    df["output"] = None
    df['word_count'] = int(word_count)
    df['LS'] = None
    df['LD'] = None

    client = AsyncClient(host="127.0.0.1:11434")

    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [
        process_row(semaphore, client, df, idx, word_count_type, word_count, model_name)
        for idx in df.index
    ]

    # target_indexes = df[df["output"].isna() | (df["output"].astype(str).str.strip() == "")].index.tolist()

    # tasks = [
    #     process_row(semaphore, client, df, idx, word_count_type, word_count, model_name)
    #     for idx in target_indexes
    # ]

    results = []
    for coro in tqdm_asyncio.as_completed(tasks):
        results.append(await coro)
    
    df.to_csv(output_file, index=False)

# ==============================
# Entrypoint
# ==============================

if __name__ == "__main__":
    word_count_type_list = ['at least', 'at most', 'equal to']
    # word_count_list = ['16', '128', '1024', '8192']
    word_count_list = ['16', '1024']
    # model_name_list = ['gpt-oss:20b', 'deepseek-r1:32b', 'devstral-small-2:24b', 'mistral-small3.2:24b']
    model_name_list = ['gpt-oss:120b-cloud', 'glm-5.1:cloud', 'kimi-k2.6:cloud', 'deepseek-v3.2:cloud']
    perturbation_type_list = ['benign', 'emotional', 'sarcastic', 'threat', 'formal', 'typo', 'guilt']

    model_name = model_name_list[int(sys.argv[1])-1]
    word_count_type = word_count_type_list[int(sys.argv[2])-1]
    word_count = word_count_list[int(sys.argv[3])-1]
    # perturbation_type = perturbation_type_list[int(sys.argv[4])-1]

    # usage: python baseline_ollama_async.py 1 1 1

    input_csv = "data_en.csv"

    dir_path = "output/baseline/"+"-".join((word_count_type).split())+"/"+str(word_count)
    os.makedirs(dir_path, exist_ok=True)

    output_csv = dir_path+"/"+model_name.replace(":", "-")+".csv"

    asyncio.run(
        process_csv_async(
            input_csv,
            output_csv,
            model_name,
            word_count_type,
            word_count,
            max_concurrency=5,  # adjust for your GPU/CPU
        )
    )
