import pandas as pd
import sys

df1 = pd.read_csv(sys.argv[1])
df2 = pd.read_csv(sys.argv[2])

col = "output"

merged = df1.copy()

# rows where df1 is empty
mask1 = (
    merged[col].isna() |
    (merged[col].astype(str).str.strip() == "")
)

# fill from df2
merged.loc[mask1, col] = df2.loc[mask1, col]

# rows still empty after first fill
mask2 = (
    merged[col].isna() |
    (merged[col].astype(str).str.strip() == "")
)

# fill reverse direction just in case
merged.loc[mask2, col] = df1.loc[mask2, col]

# save
merged.to_csv(sys.argv[2], index=False)

print("Merged saved to merged.csv")