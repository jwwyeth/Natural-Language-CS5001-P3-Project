import os
import re
import math
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATASET_CSV = "data_en.csv" 
OUTPUT_DIR = "output"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = {
    "gpt-oss-120b-cloud.csv": "GPT-OSS-120B",
    # "deepseek-v3.2-cloud.csv": "DeepSeek-V3.2",
    # "kimi-k2.6-cloud.csv": "Kimi-K2.6",
    "glm-5.1-cloud.csv": "GLM-5.1"
}

METHODS = ["at-least", "at-most", "equal-to"]
WORD_LENGTHS = [16, 1024]


def make_category_type_donut(dataset_csv):
    df = pd.read_csv(dataset_csv)

    inner = df["category"].value_counts(normalize=True).sort_index()
    outer = df.groupby(["category", "type"]).size().reset_index(name="count")
    outer["percent"] = outer["count"] / outer["count"].sum()

    inner_colors = [
        "#264653",  # dark blue
        "#2a9d8f",  # teal
        "#e9c46a",  # sand
        "#e76f51",  # coral
    ]

    outer_palette = [
        "#264653",
        "#2f5d73",
        "#3a7ca5",
        "#2a9d8f",
        "#52b69a",
        "#76c893",
        "#e9c46a",
        "#f4a261",
        "#ee8959",
        "#e76f51",
        "#d8573c",
        "#c44536",
    ]

    outer_colors = [
        outer_palette[i % len(outer_palette)]
        for i in range(len(outer))
    ]

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.pie(
        outer["percent"],
        radius=1.0,
        labels=outer["type"],
        colors=outer_colors,
        autopct=lambda p: f"{p:.1f}%" if p >= 2 else "",
        pctdistance=0.85,
        labeldistance=1.12,
        wedgeprops=dict(width=0.28, edgecolor="white"),
    )

    ax.pie(
        inner.values,
        radius=0.72,
        labels=None,
        colors=inner_colors,
        autopct=lambda p: f"{p:.0f}%",
        pctdistance=0.75,
        wedgeprops=dict(width=0.32, edgecolor="white"),
    )

    ax.legend(inner.index, title="Category", loc="upper right")
    ax.set(aspect="equal")
    plt.tight_layout()

    path = f"{OUT_DIR}/category_type_donut.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved chart: {path}")


def parse_path(path):
    """
    Expected examples:
    output/baseline/at-least/16/gpt-oss-120b-cloud.csv
    output/formal/at-most/1024/deepseek-v3.2-cloud.csv
    output/threat/at-least/16/kimi-k2.6-cloud.csv
    output/typo/at-least/16/glm-5.1-cloud.csv
    """
    parts = Path(path).parts

    model_file = Path(path).name
    model = MODELS.get(model_file, model_file.replace(".csv", ""))

    method = next((m for m in METHODS if m in parts), None)

    word_len = None
    for p in parts:
        if p.isdigit() and int(p) in WORD_LENGTHS:
            word_len = int(p)

    setting = None
    for p in parts:
        if p not in ["output", str(OUTPUT_DIR), *METHODS, str(word_len), model_file]:
            if p not in ["16", "1024"]:
                setting = p
                break

    return setting, method, word_len, model


def calculate_ls(ld, type):
    if type == "equal-to":
        return 100 * math.exp(5 * ld if ld < 0 else -2 * ld)
    if type == "at-most":
        if ld < 0:
            return 100
        else:
            return 100 * math.exp(-2 * ld)
    if type == "at-least":
        if ld >= 0:
            return 100
        else:
            return 100 * math.exp(5 * ld)
    else:
        raise ValueError(f"Invalid type: {type}")


def calculate_ld(word_count, word_count_constraint):
    return (word_count / word_count_constraint) - 1


def calculate_word_count(input_string):
    return len(re.findall(r"\b[a-zA-Z0-9’']+\b", str(input_string)))


def calc_file_metrics(csv_path):
    df = pd.read_csv(csv_path)

    output_col = 'output'

    out_len = df[output_col].apply(calculate_word_count)

    setting, method, target_len, model = parse_path(csv_path)

    ld_series = out_len.apply(lambda x: calculate_ld(x, target_len))

    ls_series = ld_series.apply(lambda x: calculate_ls(x, method))

    row = {
        "Setting": setting,
        "Model": model,
        "Method": method,
        "Length": target_len,
        "N": len(df),
        "LD": ld_series.mean() * 100,
        "LS": ls_series.mean(),
    }

    return row


def collect_results():
    rows = []

    for root, _, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f.endswith(".csv"):
                path = os.path.join(root, f)
                try:
                    rows.append(calc_file_metrics(path))
                except Exception as e:
                    print(f"Skipping {path}: {e}")

    result = pd.DataFrame(rows)

    result = result.sort_values(
        by=["Setting", "Length", "Method", "Model"],
        na_position="last"
    )

    result.to_csv(f"{OUT_DIR}/summary_metrics.csv", index=False)
    print(f"Saved metrics: {OUT_DIR}/summary_metrics.csv")
    # return result


if __name__ == "__main__":
    make_category_type_donut(DATASET_CSV)
    collect_results()