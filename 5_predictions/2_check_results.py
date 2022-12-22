# %%
from IPython.display import HTML
from pathlib import Path
import pandas as pd

import sys
sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

from lib.ml import compute_ci

pd.set_option("display.max_columns", None)
# %%
results_dir = Path("./results/1_max_data")
out_dir = Path("./summary/1_max_data")
out_dir.mkdir(exist_ok=True, parents=True)

all_df = []
for fname in results_dir.glob("*.csv"):
    df = pd.read_csv(fname, sep=";", index_col=0)
    if fname.name.endswith("mc.csv"):
        df["cv"] = "mc"
    elif fname.name.endswith("kfold.csv"):
        df["cv"] = "kfold"
    else:
        df["cv"] = "loo"
    all_df.append(df)

# %%
df = pd.concat(all_df)
# %%

to_group = ["features", "model", "target", "cv"]
no_loo_cv = df.query("cv != 'loo'").groupby(to_group + ["repeat"]).mean()
summary_df = no_loo_cv.groupby(to_group).agg(compute_ci)


# %%
summary_df.to_csv(out_dir / "summary.csv", sep=";")
df.to_csv(out_dir / "full.csv", sep=";")
# %%
HTML(summary_df.to_html())
# %%
to_print = summary_df.query("target == 'combined.outcome.3' and model == 'gsrf' and cv == 'kfold' and features == 'fmri'")
HTML(to_print.to_html())
print(to_print)
# %%

df_loo = df.query("cv == 'loo'")[
    ["features", "model", "target", "fold", "test_accuracy"]]

df_loo_summary = df_loo.groupby(["features", "model", "target"]).mean()

df_loo_summary.to_csv(out_dir / "loo_summary.csv", sep=";")