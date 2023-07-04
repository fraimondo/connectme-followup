# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from julearn.viz import plot_scores

import sys

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

from lib.ml import compute_ci
from lib.constants import FEATURE_SET_TO_NAME

# %%


cv = "kfold"
kind = "3_same_data"

t_target = "combined.outcome.3"
# t_target = "combined.outcome.12"
t_model = "gssvm"
# t_model = "gsrf"

results_dir = Path("../5_predictions/results/") / kind

results_file = results_dir.glob(f"*_{t_model}_{t_target}_{cv}.csv")

all_results = []
for t_fname in results_file:
    t_df = pd.read_csv(t_fname, sep=";", index_col=0)
    if t_df["features"][0] == "abcd_model":
        continue
    t_df["model"] = t_df["features"]
    print(f"Read {t_fname}")
    print(f"\t{t_df['features'].unique()}")
    all_results.append(t_df)
# df_result = pd.read_csv(results_dir / "full.csv", sep=";", index_col=0)

# Fix this as we computed using an older version of julearn
# df_result["cv_mdsum"] = "comparable"
# df_result["n_train"] = 46
# df_result["n_test"] = 12

# %%

# all_scores = []
# for t_feature in df_result["features"].unique():
#     t_df = df_result.query("features == @t_feature")
#     t_df["model"] = t_feature
#     all_scores.append(t_df)

# %%
panel = plot_scores(*all_results)
panel.servable()
