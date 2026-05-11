import os
import re
import math
import colorsys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


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
RAW_LENGTH_ROWS = []


def adjust_lightness(color, amount=1.0):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    l = max(0, min(1, amount * l))

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)


def make_category_type_donut(dataset_csv):
    df = pd.read_csv(dataset_csv)

    inner = (
        df["category"]
        .value_counts(normalize=True)
        .sort_index()
    )

    outer = (
        df.groupby(["category", "type"])
        .size()
        .reset_index(name="count")
    )

    outer["percent"] = (
        outer["count"] / outer["count"].sum()
    )

    # base category colors
    palette = [
        "#264653",
        "#2a9d8f",
        "#e9c46a",
        "#e76f51",
        "#3a7ca5",
        "#c44536",
    ]

    categories = list(inner.index)

    category_colors = {
        cat: palette[i % len(palette)]
        for i, cat in enumerate(categories)
    }

    # inner ring colors
    inner_colors = [
        category_colors[c]
        for c in inner.index
    ]

    # outer ring = shades of parent category
    outer_colors = []

    for category in outer["category"]:

        base = category_colors[category]

        same_cat_count = (
            outer["category"] == category
        ).sum()

        same_cat_seen = (
            outer.iloc[:len(outer_colors)]["category"] == category
        ).sum()

        if same_cat_count == 1:
            factor = 1.0
        else:
            factor = 0.7 + (
                0.6 * same_cat_seen / (same_cat_count - 1)
            )

        outer_colors.append(
            adjust_lightness(base, factor)
        )

    fig, ax = plt.subplots(figsize=(10, 10))

    # outer ring
    ax.pie(
        outer["percent"],
        radius=1.0,
        labels=outer["type"],
        colors=outer_colors,
        autopct=lambda p: f"{p:.1f}%" if p >= 2 else "",
        pctdistance=0.86,
        labeldistance=1.10,
        wedgeprops=dict(
            width=0.28,
            edgecolor="white"
        ),
    )

    # inner ring
    ax.pie(
        inner.values,
        radius=0.72,
        labels=None,
        colors=inner_colors,
        autopct=lambda p: f"{p:.0f}%",
        pctdistance=0.75,
        wedgeprops=dict(
            width=0.32,
            edgecolor="white"
        ),
    )

    legend_handles = [
        Patch(
            facecolor=category_colors[c],
            label=c
        )
        for c in inner.index
    ]

    ax.legend(
        handles=legend_handles,
        title="Category",
        loc="upper right"
    )

    ax.set(aspect="equal")

    plt.tight_layout()

    path = f"{OUT_DIR}/category_type_donut.pdf"

    plt.savefig(
        path,
        bbox_inches="tight"
    )

    plt.savefig(
        path.replace(".pdf", ".png"),
        dpi=300,
        bbox_inches="tight"
    )

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

    target_indexes = df[df[output_col].isna() | (df[output_col].astype(str).str.strip() == "")].index.tolist()

    df = df.drop(index=target_indexes)

    out_len = df[output_col].apply(calculate_word_count)

    setting, method, target_len, model = parse_path(csv_path)

    ld_series = out_len.apply(lambda x: calculate_ld(x, target_len))

    ls_series = ld_series.apply(lambda x: calculate_ls(x, method))

    for x in out_len:
        RAW_LENGTH_ROWS.append({
            "Setting": setting,
            "Model": model,
            "Method": method,
            "Length": target_len,
            "OutputLength": x,
        })

    row = {
        "Setting": setting,
        "Model": model,
        "Method": method,
        "Length": target_len,
        "Mean": out_len.mean(),
        # "N": len(df),
        'Empty': len(target_indexes),
        "LD": ld_series.mean() * 100,
        "LS": ls_series.mean(),
    }

    return row


def make_violin_grid(raw_df):
    settings = ["baseline", "typo", "formal", "threat"]
    models = list(MODELS.values())

    for method in METHODS:
        for target in WORD_LENGTHS:

            fig, ax = plt.subplots(figsize=(14, 6))

            data = []
            labels = []
            positions = []

            pos = 1

            for setting in settings:
                for model in models:

                    values = raw_df[
                        (raw_df["Setting"] == setting) &
                        (raw_df["Model"] == model) &
                        (raw_df["Method"] == method) &
                        (raw_df["Length"] == target)
                    ]["OutputLength"].tolist()

                    if values:
                        data.append(values)
                        labels.append(f"{setting}\n{model}")
                        positions.append(pos)

                    pos += 1

                pos += 0.7  # spacing between settings

            if data:
                ax.violinplot(
                    data,
                    positions=positions,
                    showmeans=True,
                    showmedians=True
                )

            ax.axhline(target, linestyle="--", linewidth=1.5)

            ax.set_title(
                f"Output Length Distribution ({method}, target={target})"
            )

            ax.set_ylabel("Generated Output Length")

            ax.set_xticks(positions)
            ax.set_xticklabels(
                labels,
                rotation=35,
                ha="right",
                fontsize=9
            )

            plt.tight_layout()

            safe_method = method.replace("-", "_")

            plt.savefig(
                f"{OUT_DIR}/violin_{safe_method}_{target}.pdf",
                bbox_inches="tight"
            )

            plt.savefig(
                f"{OUT_DIR}/violin_{safe_method}_{target}.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()


def collect_results():
    rows = []

    for root, _, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f.endswith(".csv") and f in MODELS:
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

    raw_df = pd.DataFrame(RAW_LENGTH_ROWS)
    raw_df.to_csv(f"{OUT_DIR}/raw_output_lengths.csv", index=False)
    print(f"Saved raw lengths: {OUT_DIR}/raw_output_lengths.csv")

    # make_distribution_charts(raw_df)
    make_violin_grid(raw_df)


if __name__ == "__main__":
    make_category_type_donut(DATASET_CSV)
    collect_results()
