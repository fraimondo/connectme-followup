# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

from lib.ml import compute_ci

# %%

kind = "1_max_data"
cv = "kfold"
# kind = "3_same_data"
results_dir = Path("../5_predictions/summary/") / kind

df_result = pd.read_csv(results_dir / "full.csv", sep=";", index_col=0)
print(df_result.head())


# %%
sns.set_context(
    "paper",
    rc={
        "font.size": 10,
        "axes.labelsize": 10,
        "lines.linewidth": 1,
        "xtick.labelsize": 8,
        "ytick.labelsize": 10,
    },
)
sns.set_style(
    "white",
    {
        "font.sans-serif": ["Helvetica"],
        "pdf.fonttype": 42,
        "axes.edgecolor": ".8",
    },
)

mpl.rcParams.update({"font.weight": "ultralight"})
sns.set_color_codes()
current_palette = sns.color_palette()

# %%
print("Folds:", df_result["fold"].unique())
print("Repeats:", df_result["repeat"].unique())
print("Models:", df_result["model"].unique())
print("Target:", df_result["target"].unique())
print("Features:", df_result["features"].unique())
print("CV:", df_result["cv"].unique())


# %%
models_to_plot = [
    "gssvm",
    "gsrf",
]
targets_to_plot = [
    "combined.outcome.3",
    "combined.outcome.12"
]
features_to_plot = [
    "visual",
    "visual_abcd",
    "model",
    "visual_abcd_resting",
    "visual_abcd_model",
    "resting_visual_abcd_model",
]
metrics_to_plot = [
    "test_roc_auc",
    "test_accuracy",
    "test_recall",
    "test_precision",
    "test_balanced_accuracy",
]

metrics_labels = {
    "test_roc_auc": "ROC AUC",
    "test_accuracy": "Accuracy",
    "test_recall": "Recall",
    "test_precision": "Precision",
    "test_balanced_accuracy": "Balanced Accuracy",
}

features_labels = {
    "visual": "VISUAL",
    "visual_abcd": "VISUAL\nABCD",
    "model": "MODEL",
    "visual_abcd_resting": "VISUAL\nABCD\nRESTING",
    "visual_abcd_model": "VISUAL\nABCD\nMODEL",
    "resting_visual_abcd_model": "RESTING\nVISUAL\nABCD\nMODEL",
}

model_labels = {
    "gssvm": "Support Vector Machine (GridSearch)",
    "gsrf": "Random Forest (GridSearch)",
}

target_labels = {
    "combined.outcome.3": "3-months",
    "combined.outcome.12": "12-months",
}

# %%
for t_model in models_to_plot:
    for t_target in targets_to_plot:
        for t_metric in metrics_to_plot:
            t_df = df_result[
                (df_result["model"] == t_model)
                & (df_result["target"] == t_target)
                & (df_result["features"].isin(features_to_plot))
                & (df_result["cv"] == cv)
            ]
            t_df = (
                t_df.groupby(
                    ["model", "features", "target", "repeat"])[metrics_to_plot]
                .mean()
                .reset_index()
            )
            fig, t_ax = plt.subplots(1, 1, figsize=(8, 5))
            sns.swarmplot(
                data=t_df,
                x="features",
                y=t_metric,
                s=3,
                alpha=0.4,
                color="k",
                ax=t_ax,
                order=features_to_plot,
            )
            sns.boxplot(
                data=t_df,
                x="features",
                y=t_metric,
                ax=t_ax,
                whis=(2.5, 97.5),
                color="white",
                showfliers=False,
                order=features_to_plot,
                conf_intervals=None
            )
            [
                t_ax.axhline(x, color="k", linestyle="--", alpha=0.4, lw=0.5)
                for x in np.arange(0, 1.01, 0.1)
            ]
            y_start = 0
            if t_metric == "test_roc_auc":
                y_start = 0.4
            t_ax.set_ylim(y_start, 1)
            t_ax.set_yticks(np.arange(y_start, 1.01, 0.1))
            t_ax.set_ylabel(metrics_labels[t_metric])
            t_ax.set_xlabel("")
            xticklabels = [features_labels[x] for x in features_to_plot]
            t_ax.set_xticklabels(xticklabels)
            fig.suptitle(
                f"{model_labels[t_model]} - "
                f"Outcome at {target_labels[t_target]}"
            )
            fig.savefig(
                f"./figs/{kind}/pdf_{t_model}_{t_target}_{t_metric}.pdf",
                bbox_inches="tight",
            )
            fig.savefig(
                f"./figs/{kind}/png_{t_model}_{t_target}_{t_metric}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

# %%
