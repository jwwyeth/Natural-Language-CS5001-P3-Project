import pandas as pd
import math
import re
from pathlib import Path

path = r"E:\Projects\Coding\final project nlp\Natural-Language-CS5001-P3-Project\code\output\typo\at-least\16\gpt-oss-120b-cloud.csv"
data = pd.read_csv(path)

word_count_type = "at-least"
word_count_value = 16

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

def calculate_ld(word_count):
    return (word_count / word_count_value) - 1

def calculate_word_count(input_string):
    return len(re.findall(r"\b[a-zA-Z0-9’]+\b", str(input_string)))

ls_values = data["LS"].copy()
ld_values = data["LD"].copy()
output_values = data["output"].copy()
word_count_values = data["word_count"].copy()

for index, row in data.iterrows():
    word_count = calculate_word_count(row["output"])
    ld = calculate_ld(word_count)
    if ld != ld_values[index]:
        print(f"Index: {index}, LD: {ld}, LD values: {ld_values[index]}")
    if word_count != word_count_values[index]:
        print(f"Index: {index}, Word Count: {word_count}, Word Count values: {word_count_values[index]}")

ls_avg = ls_values.mean()
print(f"LS average: {ls_avg}")
print(f"LD average: {ld_values.mean()}")
ls_recalculated = ld_values.apply(lambda x: calculate_ls(x, word_count_type))
ls_recalculated_avg = ls_recalculated.mean()

print(f"LS recalculated average: {ls_recalculated_avg}")
print(f"Difference: {ls_avg - ls_recalculated_avg}")

# for index, row in data.iterrows():
#     current_ls = row["LS"]
#     recalculated_ls = calculate_ls(row["LD"], word_count_type)
#     difference = current_ls - recalculated_ls
#     print(f"Index: {index}, Current LS: {current_ls}, Recalculated LS: {recalculated_ls}, LD: {row['LD']}, Word Count: {row['word_count']}, Difference: {difference}")



