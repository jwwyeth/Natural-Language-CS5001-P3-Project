# Natural-Language-CS5001-P3-Project

## Description

This is the codebase for our CS5001 Project (P3) focusing on benchmarking LLMs for requirement robustness. The project involves using Ollama for asynchronous processing of natural language data, along with various tools for annotation, analysis, and results aggregation.


## Installation

1. Clone the repository:
   ```
   git clone [repository URL]
   cd Natural-Language-CS5001-P3-Project
   ```

2. Set up the virtual environment:
   ```
   python -m venv env
   # On Windows:
   env\Scripts\activate
   # On macOS/Linux:
   source env/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r code/requirements.txt
   ```

4. Install Ollama:
   Follow the instructions on the [Ollama website](https://docs.ollama.com/quickstart) to install Ollama on your system.

## Usage

### Main Scripts

**ollama_async.py** is the main script for running the asynchronous processing of the perturbed data using Ollama. It reads the perturbed data files, processes them, and outputs results to a json file.
```
py code/ollama_async.py {model}[1,2] {word_type}[1,2,3] {word_count}[1,2] {perterbation_type}[4,5,6]
```
- model: 1 for gpt-oss:120b, 2 for glm5.1
- word_type: 1 for "less than", 2 for "greater than", 3 for "equal to"
- word_count: 1 for 16 words, 2 for 1024 words
- perturbation_type: 4 for "threat", 5 for "formal", 6 for "typo"

**ollama_cloud_annotation_async.py** is the script for perturbing the LIFEBENCH dataset and judging the perturbations. It generates perturbed data files in the annotation directory.
```
py code/ollama_cloud_annotation_async.py {perturbation_type}[1:7]- While the script has 7 types of perturbations, we only used 4-6, which are "threat", "formal", and "typo".
```

**results_agr.py** is the script for aggregating results and computing metrics. It reads the output json files from the output directory, computes metrics, and saves the final results in the results directory.
```
py code/results_agr.py
```

## Additional Scripts

**run_left.py** is a script to fill in missing data from ollama_async.py, usually from timeouts. It processes the remaining data and outputs results to a json file.
Same usage as ollama_async.py

**ren_patch.py** is a bulk-processing file for run_left.py, which runs multiple instances of run_left.py with different parameters to fill in all missing data.
Usage: py code/ren_patch.py

The baseline_scripts directory contains scripts for baseline processing, which can be used for comparison with the perturbed data results. Same usage as ollama_async.py

The bulk_processing directory contains scripts for bulk processing of data, which can be used to run multiple instances of ollama_async.py with different parameters to process all the perturbed data files. Same usage as ren_patch.py


## Project Structure

```
.
├── code/             # Main source code
│   ├── annotation/                         # Perturbed data files
│   ├── baseline_scripts/                   # Scripts for baseline processing
│   ├── bulk_processing/                    # Scripts for bulk processing of data (running multiple instances of ollama_async.py with different parameters)
│   ├── output/                             # Outputs from ollama_async.py, sorted by perturbation type, word count, and condition
│   ├── results/                            # Final results and metrics
│   ├── ollama_async.py                     # Runs the Ollama asynchronous processing on the perturbed data
│   ├── ollama_cloud_annotation_async.py    # Perturbs the LIFEBENCH dataset and judges the perturbations 
│   ├── results_agr.py                      # Aggregates results and computes metrics
│   ├── data_en.csv                         # LIFEBENCH dataset in English
│   ├── requirements.txt                    # Python dependencies
│   ├── run_left.py                         # Script to fill in missing data from ollama_async.py (usually from timeouts)
│   └── ren_patch.py                        # Bulk-processing file for run_left.py
└── Paper_Research/   # Research papers or documentation
```

## Dependencies

- Python 3.11
- Ollama
- Pandas
- NumPy
- Matplotlib
- Tqdm

See `code/requirements.txt` for the full list.

## Paper_Research

The `Paper_Research` directory contains research papers and documentation related to the project, including relavant literature and assignment instructions