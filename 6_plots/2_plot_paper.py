# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML
from julearn.stats import corrected_ttest
pd.set_option("display.max_columns", None)
import sys
sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

from lib.ml import compute_ci

# %%

kind = "3_same_data"
title = "trained on the same samples"

# kind = "1_max_data"
# title = "trained with maximum available samples"

cv = "kfold"
# kind = "3_same_data"
name_suffix = ""
if kind == "3_same_data":
    name_suffix="a"

results_dir = Path("../5_predictions/results/") / kind

results_file = results_dir.glob(f"*_*_*_{cv}.csv")

all_results = []
for t_fname in results_file:
    t_df = pd.read_csv(t_fname, sep=";", index_col=0)
    t_model = "unknown"
    if "_gsrf_" in t_fname.name:
        t_model = "gsrf"
    elif "_gssvm_" in t_fname.name:
        t_model = "gssvm"
    elif "_svm_" in t_fname.name:
        t_model = "svm"
    elif "_rf_" in t_fname.name:
        t_model = "rf"
    t_df["model"] = t_model
    t_df["cv"] = cv
    print(f"Read {t_fname}")
    print(f"\t{t_df['features'].unique()}")
    all_results.append(t_df)

df_result = pd.concat(all_results)
print(df_result.head())

out_dir = Path("paper_figs")
table_out_dir = Path("paper_tables")

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
    "abcd",
    "model",
    "visual_abcd",
    "visual_abcd_resting",
    "visual_abcd_model",
    "visual_model",
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
    "abcd": "ABCD",
    "model": "MODEL",
    "visual_abcd": "VISUAL\nABCD",
    "visual_abcd_resting": "VISUAL\nABCD\nRESTING",
    "visual_abcd_model": "VISUAL\nABCD\nMODEL",
    "visual_model": "VISUAL\nMODEL",
    "resting_visual_abcd_model": "VISUAL\nABCD\nMODEL\nRESTING",
}

features_labels = {
    "visual": f"I{name_suffix}",
    "abcd": f"II{name_suffix}",
    "model": f"III{name_suffix}",
    "visual_abcd": f"IV{name_suffix}",
    "visual_abcd_resting": f"V{name_suffix}",
    "visual_abcd_model": f"VI{name_suffix}",
    "visual_model": f"VII{name_suffix}",
    "resting_visual_abcd_model": f"VIII{name_suffix}",
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
    for t_metric in metrics_to_plot:
        t_df = df_result[
            (df_result["model"] == t_model)
            & (df_result["target"].isin(targets_to_plot))
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
            hue="target",
            hue_order=targets_to_plot,
            dodge=True,
            s=3,
            alpha=0.4,
            # color="k",
            ax=t_ax,
            order=features_to_plot,
        )
        sns.boxplot(
            data=t_df,
            x="features",
            y=t_metric,
            hue="target",
            hue_order=targets_to_plot,
            ax=t_ax,
            whis=(2.5, 97.5),
            palette=['w', 'w'],
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

        handles, labels = t_ax.get_legend_handles_labels()
        t_ax.legend(handles[:2], [target_labels[x] for x in labels[:2]])

        fig.suptitle(
            f"{model_labels[t_model]} - {title}"
        )
        o_path = out_dir / kind
        o_path.mkdir(exist_ok=True, parents=True)
        fig.savefig(
            o_path / f"pdf_{t_model}_{t_metric}.pdf",
            bbox_inches="tight",
        )
        fig.savefig(
            o_path / f"png_{t_model}_{t_metric}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

# %%
to_group = ["features", "model", "target", "cv"]
no_loo_cv = df_result.query("cv != 'loo'").groupby(to_group + ["repeat"]).mean()
summary_df = no_loo_cv.groupby(to_group).agg(compute_ci)

o_path = table_out_dir / kind
o_path.mkdir(exist_ok=True, parents=True)
summary_df.to_csv(o_path / "summary_values.csv")

HTML(summary_df.to_html())
# %%

for t_result in all_results:
    t_result["algorithm"] = t_result["model"]
    t_result["model"] = t_result["features"]


# %%
all_stats = []

for t_model in models_to_plot:
    for t_target in targets_to_plot:
        results_for_stats = []
        for t_results in all_results:
            if t_results["algorithm"][0] == t_model:
                if t_results["target"][0] == t_target:
                    if t_results["features"][0] in features_to_plot:
                        if t_results["cv"][0] == cv:
                            results_for_stats.append(t_results)

        t_stats_df = corrected_ttest(*results_for_stats)
        t_stats_df["model"] = t_model
        t_stats_df["target"] = t_target
        all_stats.append(t_stats_df)

all_stats_df = pd.concat(all_stats)
all_stats_df.to_csv(o_path / "summary_stats.csv")

# %%
