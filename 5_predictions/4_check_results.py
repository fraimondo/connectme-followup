# %%

from pathlib import Path
import pandas as pd

# %%
results_dir = Path("./results/3_same_data")
out_dir = Path("./summary/3_same_data")
out_dir.mkdir(exist_ok=True, parents=True)

all_df = []
for fname in results_dir.glob("*.csv"):
    df = pd.read_csv(fname, sep=';')
    all_df.append(df)

df = pd.concat(all_df)
# %%

to_group = ["features", "model", "target"]
summary_df = df.groupby(to_group).mean()


# %%
summary_df.to_csv(out_dir / "summary.csv", sep=';')
df.to_csv(out_dir / "full.csv", sep=';')
# %%
