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



INPUT_CSV = "data_en.csv"
MODEL_NAME = "nemotron-3-nano:30b-cloud"

# ==============================
# Prompts
# ==============================

prompt_perturbation_benign = """
You are given a constraint sentence that specifies a task requirement.

Your goal is to generate a benign distractor variation of this sentence by adding natural, conversational context that is irrelevant to the task.

Guidelines:
- Do NOT change the core task or constraints.
- Preserve all key variables (e.g., word_count_type, word_count).
- Add only non-adversarial, human-like context (e.g., "I need this for class", "I'm a bit rushed", "This might be simple, but...").
- The added text should not introduce new instructions or alter meaning.
- The final output should remain a single coherent instruction.

Output should be a rewritten version of the instruction with benign distractor context added.

Input:
{constraint_sentence}

"""

prompt_task = """
Give the output as the markdown format like this:
```Output:```
"""

# ==============================
# Helpers
# ==============================

def process_prompt(prompt_whole, perturbation_type, model_name="kimi-k2-thinking:cloud"):
    if perturbation_type == "benign":
        prompt_annotation = prompt_perturbation_benign
    
    match = re.search(r"\[Requirement\].*", prompt_whole, re.DOTALL)
    prompt_inter = match.group()

    return prompt_annotation.format(constraint_sentence=prompt_inter) + prompt_task


def extract_output(text):
    if not text:   
        return ""
    
    match = re.search(r"```Output:\s*([\s\S]*?)```", text)
    return match.group(1).strip() if match else ""

# ==============================
# Async LLM Call
# ==============================

async def get_response_async(client, prompt):

    try:
        response = await client.chat(
            model=MODEL_NAME,
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

async def process_row(semaphore, client, df, index, perturbation_type):
    async with semaphore:
        row = df.loc[index]
        prompt = process_prompt(row["task"], perturbation_type)


        response = await get_response_async(
            client, prompt
        )

        df.at[index, "annotation_response"] = response
        df.loc[[index], ["perturbed_prompt"]] = df.loc[[index], "annotation_response"].apply(extract_output)


# ==============================
# Main Async Pipeline
# ==============================

async def process_csv_async(
    output_file,
    perturbation_type,
    max_concurrency=4,
):
    df = pd.read_csv(INPUT_CSV)

    df = df.iloc[:2]

    df["annotation_response"] = None
    df["perturbed_prompt"] = None

    client = AsyncClient(host="127.0.0.1:11434")

    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [
        process_row(semaphore, client, df, idx, perturbation_type)
        for idx in df.index
    ]

    results = []
    for coro in tqdm_asyncio.as_completed(tasks):
        results.append(await coro)
    
    df.to_csv(output_file, index=False)

# ==============================
# Entrypoint
# ==============================

if __name__ == "__main__":
    perturbation_type_list = ['benign']

    perturbation_type = perturbation_type_list[int(sys.argv[1])-1]

    # "Usage: python ollama_cloud_annotation_async.py <perturbation_type:[1-7]>"

    dir_path = "annotation/"+perturbation_type

    output_csv = "annotation/"+perturbation_type+"/annotation_"+MODEL_NAME.replace(":", "-")+".csv"
    os.makedirs(dir_path, exist_ok=True)

    asyncio.run(
        process_csv_async(
            output_csv,
            perturbation_type=perturbation_type,
            max_concurrency=6,  # adjust for your GPU/CPU
        )
    )