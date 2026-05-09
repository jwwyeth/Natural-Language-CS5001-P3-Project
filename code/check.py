import pandas as pd
import sys

INPUT_CSV = sys.argv[1]
df = pd.read_csv(INPUT_CSV)
# print((df['word_count'] > 16).sum())
idx = df.index[df['perturbed_prompt'].isna()]
# idx = df.index[df['output'].isna()]
idx_list = idx.tolist()
print(df.shape)
print(len(idx_list), idx_list)