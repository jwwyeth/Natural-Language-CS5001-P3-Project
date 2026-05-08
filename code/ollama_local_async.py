import os
import os
import re
import sys
import math
import time
import asyncio
import warnings
import traceback
import subprocess
import pandas as pd
from ollama import AsyncClient
from tqdm.asyncio import tqdm_asyncio


warnings.filterwarnings("ignore")

# ==============================
# Constants
# ==============================

OLLAMA_TIMEOUT_SECONDS = 300  # 5 minutes
MODEL_RESTART_DELAY = 10  # seconds to wait after stopping model

MODEL_DIR="/share/ceph/scratch/gas2bt/ollama_models"
CONTAINER="ollama_container"
# ==============================
# Global State
# ==============================

model_stopped_event = None  # Event to signal when model is stopped due to timeout
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

def process_prompt(perturbed_prompt, raw_data, word_count_type, word_count):
    perturbed_prompt = "" if pd.isna(perturbed_prompt) else str(perturbed_prompt)

    prompt_final = (
        perturbed_prompt
        .replace("{word_count_type}", str(word_count_type))
        .replace("{word_count}", str(word_count))
    )
    
    return prompt_execution.format(perturbed_prompt=prompt_final, raw_data=raw_data) + prompt_task

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

def kill_ollama_process(model_name):
    """Stop the Ollama model using apptainer exec command"""
    global model_stopped_event
    try:
        print("\n⚠️  Timeout reached! Stopping Ollama model...")
        
        # Construct apptainer command
        cmd = [
            "apptainer", "exec", "--nv",
            "--bind", f"{MODEL_DIR}:/models",
            "--env", "OLLAMA_MODELS=/models",
            CONTAINER,
            "ollama", "stop", model_name
        ]
        
        subprocess.run(cmd, capture_output=True, timeout=5)
        print(f"✓ Ollama model '{model_name}' stopped")
        
        # Signal that model has been stopped - this will cancel other requests
        if model_stopped_event:
            model_stopped_event.set()
            
        # Wait a moment for the process to fully stop
        time.sleep(2)
    except Exception as e:
        print(f"Error stopping Ollama model: {e}")

# ==============================
# Async LLM Call
# ==============================

async def get_response_async(client, model_name, prompt):
    """Get response from Ollama with timeout handling"""
    global model_stopped_event
    
    try:
        # Check if model was stopped by another request
        if model_stopped_event and model_stopped_event.is_set():
            print(f"⚠️  Model '{model_name}' was stopped due to timeout in another request")
            raise asyncio.CancelledError("Model stopped due to timeout")
            
        response = await asyncio.wait_for(
            client.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_ctx": 65536},
            ),
            timeout=OLLAMA_TIMEOUT_SECONDS
        )
        return response["message"]["content"]
            
    except asyncio.TimeoutError:
        print(f"⚠️  Request timeout after {OLLAMA_TIMEOUT_SECONDS} seconds")
        kill_ollama_process(model_name)
        raise
    except asyncio.CancelledError:
        # Model was stopped by another request - re-raise to trigger retry
        raise
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
        prompt = process_prompt(row["perturbed_prompt"], row["raw_data"], word_count_type, word_count)

        limit = 3
        timeout_occurred = False
        
        for attempt in range(limit):
            try:
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
                else:
                    break  # Success, exit retry loop
                    
            except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                error_type = "TIMEOUT" if isinstance(e, asyncio.TimeoutError) else "MODEL_STOPPED"
                print(f"Attempt {attempt+1}/{limit}: {error_type} for index {index}")
                
                # If this is the first time we've detected a model stop, restart it
                if isinstance(e, asyncio.TimeoutError) or (model_stopped_event and model_stopped_event.is_set()):
                    if not timeout_occurred:  # Only restart once per batch of failures
                        print(f"🔄 Attempting to restart model '{model_name}'...")
                        # Clear the stopped event to allow new requests
                        if model_stopped_event:
                            model_stopped_event.clear()
                        timeout_occurred = True
                
                # Wait for model to be ready before retrying
                await asyncio.sleep(MODEL_RESTART_DELAY)
                print(f"Retrying index {index}...")
                continue
            except Exception as e:
                print(f"Attempt {attempt+1}/{limit} error for index {index}: {e}")
                await asyncio.sleep(2)

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
    global model_stopped_event
    
    # Initialize the global event for this processing run
    model_stopped_event = asyncio.Event()
    
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
    model_name_list = ['gpt-oss:20b', 'deepseek-r1:32b', 'devstral-small-2:24b', 'mistral-small3.2:24b']
    perturbation_type_list = ['benign', 'emotional', 'sarcastic', 'threat', 'formal', 'typo', 'guilt']

    model_name = model_name_list[int(sys.argv[1])-1]
    word_count_type = word_count_type_list[int(sys.argv[2])-1]
    word_count = word_count_list[int(sys.argv[3])-1]
    perturbation_type = perturbation_type_list[int(sys.argv[4])-1]

    input_csv = "annotation/"+perturbation_type+"_qwen3.6-latest.csv"

    # "Usage: ollama_cloud_async.py <model_name:[1-4]> <word_count_type:[1-3]> <word_count:[1,2]> <perturbation_type:[1-7]>"

    dir_path = "output/"+perturbation_type+"/"+"-".join((word_count_type).split())+"/"+str(word_count)
    os.makedirs(dir_path, exist_ok=True)

    output_csv = dir_path+"/"+model_name.replace(":", "-")+".csv"

    asyncio.run(
        process_csv_async(
            input_csv,
            output_csv,
            model_name,
            word_count_type,
            word_count,
            max_concurrency=6,  # adjust for your GPU/CPU
        )
    )
