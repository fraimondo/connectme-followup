# %%
from IPython.display import HTML
from pathlib import Path
import pandas as pd

import sys
sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

from lib.ml import compute_ci

pd.set_option("display.max_columns", None)
# %%
results_dir = Path("./results/3_same_data")
out_dir = Path("./summary/3_same_data")
out_dir.mkdir(exist_ok=True, parents=True)

all_df = []
for fname in results_dir.glob("*.csv"):
    df = pd.read_csv(fname, sep=";", index_col=0)
    if fname.name.endswith("mc.csv"):
        df["cv"] = "mc"
    elif fname.name.endswith("kfold.csv"):
        df["cv"] = "kfold"
    else:
        df["cv"] = "wrong"
    all_df.append(df)

# %%
df = pd.concat(all_df)
# %%

to_group = ["features", "model", "target", "cv"]
kfold_cv = df.groupby(to_group + ["repeat"]).mean()
summary_df = kfold_cv.groupby(to_group).agg(compute_ci)


# %%
summary_df.to_csv(out_dir / "summary.csv", sep=";")
df.to_csv(out_dir / "full.csv", sep=";")
# %%
HTML(summary_df.to_html())

# %%
