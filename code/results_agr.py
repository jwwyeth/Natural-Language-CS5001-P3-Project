import os
import re
import math
import colorsys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rouge_score import rouge_scorer
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

    target_indexes = df[
        df[output_col].isna() |
        (df[output_col].astype(str).str.strip() == "")
    ].index.tolist()

    df = df.drop(index=target_indexes)

    out_len = df[output_col].apply(calculate_word_count)

    setting, method, target_len, model = parse_path(csv_path)

    ld_series = out_len.apply(
        lambda x: calculate_ld(x, target_len)
    )

    ls_series = ld_series.apply(
        lambda x: calculate_ls(x, method)
    )

    # PASS / FAIL
    if method == "equal-to":
        pass_mask = out_len == target_len

    elif method == "at-most":
        pass_mask = out_len <= target_len

    elif method == "at-least":
        pass_mask = out_len >= target_len

    else:
        raise ValueError(f"Unknown method: {method}")

    pass_count = pass_mask.sum()
    fail_count = len(out_len) - pass_count

    pass_ratio = pass_count / len(out_len)
    fail_ratio = fail_count / len(out_len)

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

        "Empty": len(target_indexes),
        "Count": len(out_len),

        "LD": ld_series.mean() * 100,
        "LS": ls_series.mean(),

        "Pass": pass_count,
        "Fail": fail_count,

        "PassRatio": pass_ratio * 100,
        "FailRatio": fail_ratio * 100,
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
            within_gap = 0.6
            group_gap = 0.2

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

                    pos += within_gap

                # pos += 0.7  # spacing between settings
                pos += group_gap

            if data:
                ax.violinplot(
                    data,
                    positions=positions,
                    showmeans=True,
                    showmedians=True
                )

            ax.set_title(f"Output Length Distribution ({method}, target={target})",
                        fontsize=20, fontweight="bold")
            ax.set_ylabel("Generated Output Length", fontsize=18)
            ax.tick_params(axis="y", labelsize=16)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=15)
            ax.axhline(target, linestyle="--", linewidth=1.5)     

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
    # raw_df.to_csv(f"{OUT_DIR}/raw_output_lengths.csv", index=False)
    # print(f"Saved raw lengths: {OUT_DIR}/raw_output_lengths.csv")

    # make_distribution_charts(raw_df)
    make_violin_grid(raw_df)


def make_ls_by_method_chart(summary_csv="results/summary_metrics.csv", out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(summary_csv)

    settings = ["baseline", "formal", "threat", "typo"]
    models = ["GPT-OSS-120B", "GLM-5.1"]
    methods = ["at-least", "at-most", "equal-to"]

    colors = {
        "baseline": "#355070",
        "formal": "#6d597a",
        "threat": "#b56576",
        "typo": "#e56b6f",
    }

    hatches = {
        "GPT-OSS-120B": "",
        "GLM-5.1": "//",
    }

    pairs = [(s, m) for s in settings for m in models]

    for method in methods:
        fig, ax = plt.subplots(figsize=(8, 4))

        x16 = np.arange(len(pairs)) * 0.14
        x1024 = x16 + 1.4
        width = 0.1

        vals16 = []
        vals1024 = []

        for setting, model in pairs:
            row16 = df[
                (df["Method"] == method) &
                (df["Setting"] == setting) &
                (df["Model"] == model) &
                (df["Length"] == 16)
            ]

            row1024 = df[
                (df["Method"] == method) &
                (df["Setting"] == setting) &
                (df["Model"] == model) &
                (df["Length"] == 1024)
            ]

            vals16.append(row16["LS"].iloc[0] if not row16.empty else np.nan)
            vals1024.append(row1024["LS"].iloc[0] if not row1024.empty else np.nan)

        for idx, ((setting, model), val) in enumerate(zip(pairs, vals16)):
            ax.bar(
                x16[idx],
                val,
                width=width,
                color=colors[setting],
                hatch=hatches[model],
                edgecolor="black",
                linewidth=0.3,
            )

        for idx, ((setting, model), val) in enumerate(zip(pairs, vals1024)):
            ax.bar(
                x1024[idx],
                val,
                width=width,
                color=colors[setting],
                hatch=hatches[model],
                edgecolor="black",
                linewidth=0.3,
            )

        ax.axhline(
            100,
            linestyle="--",
            linewidth=1.0,
            color="black"
        )

        ax.axvline(
            (x16[-1] + x1024[0]) / 2,
            color="gray",
            linestyle=":",
            linewidth=1.0
        )

        ax.set_ylim(75, 101)
        ax.set_xticks([])

        ax.text(
            np.mean(x16),
            73.7,
            "Target Length = 16",
            ha="center",
            fontsize=11,
            # fontweight="bold"
        )

        ax.text(
            np.mean(x1024),
            73.7,
            "Target Length = 1024",
            ha="center",
            fontsize=11,
            # fontweight="bold"
        )

        # ax.set_title(
        #     f"{method.replace('-', ' ').title()} Constraint",
        #     fontsize=15,
        #     fontweight="bold",
        #     pad=12
        # )

        ax.set_ylabel("Length Satisfaction (LS)", fontsize=12)
        ax.tick_params(axis="y", labelsize=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.6,
            alpha=0.45
        )

        legend_handles = [
            Patch(facecolor=colors[s], edgecolor="black", label=s)
            for s in settings
        ]

        legend_handles += [
            Patch(facecolor="white", edgecolor="black", hatch="", label="GPT-OSS-120B"),
            Patch(facecolor="white", edgecolor="black", hatch="//", label="GLM-5.1"),
        ]

        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=6,
            fontsize=9,
            frameon=False,
            columnspacing=0.9,
            handlelength=1.2,
            handletextpad=0.4
        )

        plt.tight_layout()

        safe_method = method.replace("-", "_")

        plt.savefig(f"{out_dir}/ls_by_method_{safe_method}.pdf", bbox_inches="tight")
        plt.savefig(f"{out_dir}/ls_by_method_{safe_method}.png", dpi=300, bbox_inches="tight")
        plt.close()


def make_short_summary(summary_csv="results/summary_metrics.csv", out_csv="results/short_pass_fail_summary.csv"):
    df = pd.read_csv(summary_csv)

    short_df = (
        df.groupby(["Setting", "Model"])
        .agg({
            "Pass": "sum",
            "Fail": "sum",
            "Empty": "sum",
            "Count": "sum"
        })
        .reset_index()
    )

    short_df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")


def make_rouge_summary(out_csv="results/rouge_summary.csv"):
    rows = []

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    for root, _, files in os.walk(OUTPUT_DIR):
        for f in files:
            if not (f.endswith(".csv") and f in MODELS):
                continue

            csv_path = os.path.join(root, f)
            setting, method, target_len, model = parse_path(csv_path)

            if setting == "baseline":
                continue

            baseline_path = os.path.join(
                OUTPUT_DIR,
                "baseline",
                method,
                str(target_len),
                f
            )

            if not os.path.exists(baseline_path):
                print(f"Missing baseline: {baseline_path}")
                continue

            cur_df = pd.read_csv(csv_path)
            base_df = pd.read_csv(baseline_path)

            output_col = "output"
            n = min(len(cur_df), len(base_df))

            r1, r2, rl = [], [], []
            skipped = 0

            for i in range(n):
                pred = cur_df.loc[i, output_col]
                ref = base_df.loc[i, output_col]

                if (
                    pd.isna(pred) or pd.isna(ref) or
                    str(pred).strip() == "" or
                    str(ref).strip() == ""
                ):
                    skipped += 1
                    continue

                scores = scorer.score(str(ref), str(pred))

                r1.append(scores["rouge1"].fmeasure)
                r2.append(scores["rouge2"].fmeasure)
                rl.append(scores["rougeL"].fmeasure)

            rows.append({
                "Setting": setting,
                "Model": model,
                "Method": method,
                "Length": target_len,
                "Count": len(r1),
                "Skipped": skipped,
                "ROUGE-1": sum(r1) / len(r1) * 100 if r1 else None,
                "ROUGE-2": sum(r2) / len(r2) * 100 if r2 else None,
                "ROUGE-L": sum(rl) / len(rl) * 100 if rl else None,
            })

    rouge_df = pd.DataFrame(rows)

    rouge_df = rouge_df.sort_values(
        by=["Setting", "Length", "Method", "Model"],
        na_position="last"
    )

    rouge_df.to_csv(out_csv, index=False)
    print(f"Saved ROUGE summary: {out_csv}")


def make_short_rouge_from_detail(
    detail_csv="results/rouge_summary.csv",
    out_csv="results/short_rouge_summary.csv"
):
    df = pd.read_csv(detail_csv)

    short_df = (
        df.groupby(["Setting", "Model"])
        .agg({
            "ROUGE-1": "mean",
            "ROUGE-2": "mean",
            "ROUGE-L": "mean",
            "Skipped": "sum",
            "Count": "sum"
        })
        .reset_index()
    )

    short_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    make_category_type_donut(DATASET_CSV)
    collect_results()
    make_ls_by_method_chart()
    make_short_summary()
    make_rouge_summary()
    make_short_rouge_from_detail()