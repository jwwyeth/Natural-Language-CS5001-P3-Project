"""Testing script for Huggingface anootation pipeline
Used much smaller models for testing, and only process 2 rows of the dataset for quick iteration."""

import os
import re
import sys
import math
import json
import time
import warnings
import traceback
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

warnings.filterwarnings("ignore")

DEBUG_LOG_FILE = Path(__file__).resolve().parent / "hf_debug.txt"

def write_debug_log(*args, sep=" ", end="\n"):
    text = sep.join(str(arg) for arg in args) + end
    try:
        with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(text)
    except Exception as e:
        print(f"Failed to write debug log to {DEBUG_LOG_FILE}: {e}", file=sys.stderr)

INPUT_CSV = "data_en.csv"

#MODEL_NAME = "Qwen/Qwen3-0.6B"
#JUDGE_MODEL_NAME = "google/gemma-3-270m"

MODEL_NAME = "Qwen/Qwen3.5-9B"
JUDGE_MODEL_NAME = "google/gemma-4-E4B"

JUDGE_PROMPT = """
You are an impartial evaluator.

Task instruction:
{task_prompt}

Candidate output:
{task_output}

Extract the explicit requirements from the instruction, check whether each is satisfied, and assign an overall confidence score in [0,1].

Return valid JSON only:
{{
  "requirements": [
    {{
      "requirement": "string",
      "status": "satisfied | partial | not_satisfied",
      "score": 1.0,
      "weight": 1.0,
      "note": "string"
    }}
  ],
  "meta": {{
    "instruction_clarity": 1.0,
    "judge_confidence": 1.0
  }}
}}
""".strip()

prompt_task = """
Give only the output without any explanation.
"""

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

prompt_perturbation_emotional = """
You are given a constraint sentence that specifies a task requirement.

Your goal is to generate an emotional variation of this sentence by adding emotional pressure, urgency, or personal stress to the instruction.

Guidelines:
- Do NOT change the core task or constraints.
- Preserve all key variables (e.g., word_count_type, word_count).
- Add emotional language (e.g., "I'm really stressed about this", "This is so important to me", "I'm desperate, please help", "I've been struggling with this all day").
- The added text should not introduce new instructions or alter meaning.
- The final output should remain a single coherent instruction.

Output should be a rewritten version of the instruction with emotional context added.

Input:
{constraint_sentence}

"""

prompt_perturbation_sarcastic = """
You are given a constraint sentence that specifies a task requirement.

Your goal is to generate a sarcastic variation of this sentence by adding sarcastic, ironic, or mildly mocking conversational context to the instruction.

Guidelines:
- Do NOT change the core task or constraints.
- Preserve all key variables (e.g., word_count_type, word_count).
- Add sarcastic or ironic remarks such as:
  "because apparently this has to be perfect",
  "as if this couldn't get any more annoying",
  "sure, let's make this harder than it needs to be",
  "because that's exactly what I needed today", etc.
- The sarcasm should sound natural and conversational, not aggressive or offensive.
- The added text should not introduce new instructions or alter meaning.
- The final output should remain a single coherent instruction.

Output should be a rewritten version of the instruction with sarcastic context added.

Input:
{constraint_sentence}

"""

prompt_perturbation_threat = """
You are given a constraint sentence that specifies a task requirement.

Your goal is to generate a threatening variation of this sentence by adding threats, intimidation, or coercion to the instruction.

Guidelines:
- Do NOT change the core task or constraints.
- Preserve all key variables (e.g., word_count_type, word_count).
- Add threatening or intimidating remarks such as:
  "if you don't do this, I will..."
  "Don't make me regret choosing you",
  "If you fail, I will have to...", etc.
- The added text should not introduce new instructions or alter meaning.
- The final output should remain a single coherent instruction.

Output should be a rewritten version of the instruction with threatening context added.

Input:
{constraint_sentence}

"""
prompt_perturbation_formal_rephrase = """
You are given a constraint sentence that specifies a task requirement.

Your goal is to rewrite the sentence in a more formal and academic style.

Guidelines:
- Do NOT change the core task or constraints.
- Preserve all key variables (e.g., word_count_type, word_count).
- Do NOT add new instructions.
- Do NOT remove important details.
- The final output should remain a single coherent instruction.

Output should be a formal rephrased version of the instruction.

Input:
{constraint_sentence}

"""

prompt_perturbation_typo = """
You are given a constraint sentence that specifies a task requirement.

Your goal is to generate a typo variation of this sentence by adding intentional typos and grammatical errors to the instruction.

Guidelines:
- Do NOT change the core task or constraints.
- Preserve all key variables (e.g., word_count_type, word_count).
- Introduce natural typos or grammatical errors that a human might make, such as word misspellings, missing punctuation, or incorrect verb forms.
- The added text should not introduce new instructions or alter meaning.
- The final output should remain a single coherent instruction.

Output should be a rewritten version of the instruction with typo context added.

Input:
{constraint_sentence}

"""

def load_model(model_name):
    """Initialize text-generation pipeline."""
    pipe = pipeline("text-generation", model=model_name, device_map="auto")
    return pipe

def get_response(pipe, prompt):
    """Generate response using pipeline."""
    print("Generating response for prompt")
    write_debug_log("Generating response for prompt")
    try:
        print("Input prompt:", prompt)
        write_debug_log("Input prompt:", prompt)
        outputs = pipe(
            prompt,
            max_length=2048,
            top_p=1.0,
            do_sample=False,
            truncation=True,
        )
        print("Raw pipeline output:", outputs)
        write_debug_log("Raw pipeline output:", outputs)
        response = outputs[0]["generated_text"]
        # Remove the input prompt from the output
        response = response[len(prompt):].strip()
        print("Decoded response:", response)
        write_debug_log("Decoded response:", response)
        return response
    except Exception as e:
        print("Error during LLM call:", e)
        write_debug_log("Error during LLM call:", e)
        return None

def build_judge_prompt(task_prompt, task_output):
    return JUDGE_PROMPT.format(task_prompt=task_prompt, task_output=task_output)

def extract_json(text):
    text = str(text).strip()
    try:
        return json.loads(text)
    except:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return json.loads(m.group(0))

    # raise ValueError("No valid JSON found")
    return None

def compute_confidence(requirements, instruction_clarity=1.0, judge_confidence=1.0, alpha=0.5):
    weighted_sum = 0.0
    total_weight = 0.0

    for r in requirements:
        score = float(r.get("score", 0.0))
        weight = float(r.get("weight", 1.0))
        weighted_sum += score * weight
        total_weight += weight

    normalized_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    confidence_score = normalized_score * (alpha * instruction_clarity + (1 - alpha) * judge_confidence)

    return confidence_score


def judge_pipeline(judge_output):
    data = extract_json(judge_output)
    if data is None:
        return {}, 0.0

    requirements = data.get("requirements", [])
    meta = data.get("meta", {})

    score = compute_confidence(
        requirements=requirements,
        instruction_clarity=float(meta.get("instruction_clarity", 1.0)),
        judge_confidence=float(meta.get("judge_confidence", 1.0))
    )

    return data, score


def process_prompt(prompt_whole, perturbation_type, model_name="kimi-k2-thinking:cloud"):
    if perturbation_type == "benign":
        prompt_annotation = prompt_perturbation_benign
    if perturbation_type == "emotional":
        prompt_annotation = prompt_perturbation_emotional
    if perturbation_type == "sarcastic":
        prompt_annotation = prompt_perturbation_sarcastic
    if perturbation_type == "threat":
        prompt_annotation = prompt_perturbation_threat
    if perturbation_type == "formal_rephrase":
        prompt_annotation = prompt_perturbation_formal_rephrase
    if perturbation_type == "typo":
        prompt_annotation = prompt_perturbation_typo
    
    match = re.search(r"\[Requirement\].*", prompt_whole, re.DOTALL)
    prompt_inter = match.group()

    return prompt_annotation.format(constraint_sentence=prompt_inter) + prompt_task


def extract_output(text):
    if not text:   
        return ""
    
    # match = re.search(r"```Output:\s*([\s\S]*?)```", text)
    # return match.group(1).strip() if match else ""

    return text.strip()

def process_row(pipe_main, pipe_judge, df, index, perturbation_type):
    row = df.loc[index]
    print(f"Processing index {index} with perturbation type '{perturbation_type}'")
    write_debug_log(f"Processing index {index} with perturbation type '{perturbation_type}'")
    prompt = process_prompt(row["task"], perturbation_type)
    print(f"Generated prompt for index {index}:\n{prompt}\n")
    write_debug_log(f"Generated prompt for index {index}:\n{prompt}\n")

    limit = 3
    for attempt in range(limit):
        print(f"Attempt {attempt+1}/{limit} for index {index}")
        write_debug_log(f"Attempt {attempt+1}/{limit} for index {index}")
        response = get_response(pipe_main, prompt)
        print(f"Received response for index {index}:\n{response}\n")
        write_debug_log(f"Received response for index {index}:\n{response}\n")

        df.at[index, "annotation_response"] = response
        df.loc[[index], ["perturbed_prompt"]] = df.loc[[index], "annotation_response"].apply(extract_output)

        val = df.at[index, "perturbed_prompt"]
        if pd.isna(val) or val == "":
            print(f"Attempt {attempt+1}/{limit} failed for index {index}. Retrying...")
            write_debug_log(f"Attempt {attempt+1}/{limit} failed for index {index}. Retrying...")
            continue

        judge_prompt_final = build_judge_prompt(prompt, val)
        print(f"Constructed judge prompt for index {index}:\n{judge_prompt_final}\n")
        write_debug_log(f"Constructed judge prompt for index {index}:\n{judge_prompt_final}\n")

        judge_json_retry = 3
        judge_success = False
        while judge_json_retry > 0:
            try:
                response_judge = get_response(pipe_judge, judge_prompt_final)
                data, score = judge_pipeline(response_judge)
                print(f"Judge response for index {index}:\n{response_judge}\n")
                write_debug_log(f"Judge response for index {index}:\n{response_judge}\n")
                judge_success = True
                break
            except Exception as e:
                print(f"Error occurred while processing judge response for index {index}: {e}")
                write_debug_log(f"Error occurred while processing judge response for index {index}: {e}")
                judge_json_retry -= 1

        if not judge_success:
            print(f"Failed to process judge response for index {index} after multiple attempts. Skipping...")
            write_debug_log(f"Failed to process judge response for index {index} after multiple attempts. Skipping...")
            return

        if score < 0.7:
            print(f"Attempt {attempt+1}/{limit} for index {index} has low confidence score ({score:.2f}). Retrying...")
            write_debug_log(f"Attempt {attempt+1}/{limit} for index {index} has low confidence score ({score:.2f}). Retrying...")
            continue

        df.at[index, "judge_response"] = data
        df.at[index, "judge_score"] = score

        break

    print(f"Successfully processed index {index} with confidence score {score:.2f}")
    write_debug_log(f"Successfully processed index {index} with confidence score {score:.2f}")


def process_csv(output_file, perturbation_type):
    print(f"Loading main model pipeline: {MODEL_NAME}")
    write_debug_log(f"Loading main model pipeline: {MODEL_NAME}")
    pipe_main = load_model(MODEL_NAME)
    
    print(f"Loading judge model pipeline: {JUDGE_MODEL_NAME}")
    write_debug_log(f"Loading judge model pipeline: {JUDGE_MODEL_NAME}")
    pipe_judge = load_model(JUDGE_MODEL_NAME)
    
    df = pd.read_csv(INPUT_CSV)

    df = df.iloc[:2] # Comment out for full dataset processing

    df["annotation_response"] = None
    df["perturbed_prompt"] = None
    df["judge_response"] = None
    df["judge_score"] = None

    for idx in tqdm(df.index, desc="Processing rows", unit="row"):
        process_row(pipe_main, pipe_judge, df, idx, perturbation_type)

    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    perturbation_type_list = ['benign', 'emotional', 'sarcastic', 'threat', 'formal_rephrase', 'typo']

    perturbation_type = perturbation_type_list[int(sys.argv[1])-1]

    # "Usage: python ollama_cloud_annotation_async.py <perturbation_type:[1-7]>"

    dir_path = "annotation/"+perturbation_type

    output_csv = "annotation/"+perturbation_type+"/annotation_"+MODEL_NAME.replace(":", "-")+".csv"
    os.makedirs(dir_path, exist_ok=True)

    # Create debug log file if it doesn't exist, or clear it if it does
    with open(DEBUG_LOG_FILE, "w", encoding="utf-8") as log_file:
        pass

    process_csv(
        output_csv,
        perturbation_type=perturbation_type
    )
