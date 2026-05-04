import os
import re
import sys
import math
import json
import time
import warnings
import traceback
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import accelerate

warnings.filterwarnings("ignore")

INPUT_CSV = "data_en.csv"
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

def get_response(tokenizer, model, prompt):
    """Generate response using Transformers model."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=2048,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from the output
        response = response[len(prompt):].strip()
        return response
    except Exception as e:
        print("Error during LLM call:", e)
        traceback.print_exc()
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

def process_row(tokenizer_main, model_main, tokenizer_judge, model_judge, df, index, perturbation_type):
    row = df.loc[index]
    prompt = process_prompt(row["task"], perturbation_type)

    limit = 3
    for attempt in range(limit):
        response = get_response(tokenizer_main, model_main, prompt)

        df.at[index, "annotation_response"] = response
        df.loc[[index], ["perturbed_prompt"]] = df.loc[[index], "annotation_response"].apply(extract_output)

        val = df.at[index, "perturbed_prompt"]
        if pd.isna(val) or val == "":
            print(f"Attempt {attempt+1}/{limit} failed for index {index}. Retrying...")
            continue

        judge_prompt_final = build_judge_prompt(prompt, val)

        judge_json_retry = 3
        judge_success = False
        while judge_json_retry > 0:
            try:
                response_judge = get_response(tokenizer_judge, model_judge, judge_prompt_final)
                data, score = judge_pipeline(response_judge)
                judge_success = True
                break
            except Exception as e:
                print(f"Error occurred while processing judge response for index {index}: {e}")
                judge_json_retry -= 1

        if not judge_success:
            print(f"Failed to process judge response for index {index} after multiple attempts. Skipping...")
            return

        if score < 0.7:
            print(f"Attempt {attempt+1}/{limit} for index {index} has low confidence score ({score:.2f}). Retrying...")
            time.sleep(2)  # wait before retrying
            continue

        df.at[index, "judge_response"] = data
        df.at[index, "judge_score"] = score

def process_csv(output_file, perturbation_type):
    print(f"Loading main model: {MODEL_NAME}")
    tokenizer_main, model_main = load_model(MODEL_NAME)
    
    print(f"Loading judge model: {JUDGE_MODEL_NAME}")
    tokenizer_judge, model_judge = load_model(JUDGE_MODEL_NAME)
    
    df = pd.read_csv(INPUT_CSV)

    df = df.iloc[:2] # Comment out for full dataset processing

    df["annotation_response"] = None
    df["perturbed_prompt"] = None
    df["judge_response"] = None
    df["judge_score"] = None

    for idx in tqdm(df.index, desc="Processing rows", unit="row"):
        process_row(tokenizer_main, model_main, tokenizer_judge, model_judge, df, idx, perturbation_type)

    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    perturbation_type_list = ['benign', 'emotional', 'sarcastic', 'threat', 'formal_rephrase', 'typo']

    perturbation_type = perturbation_type_list[int(sys.argv[1])-1]

    # "Usage: python ollama_cloud_annotation_async.py <perturbation_type:[1-7]>"

    dir_path = "annotation/"+perturbation_type

    output_csv = "annotation/"+perturbation_type+"/annotation_"+MODEL_NAME.replace(":", "-")+".csv"
    os.makedirs(dir_path, exist_ok=True)

    process_csv(
        output_csv,
        perturbation_type=perturbation_type
    )
