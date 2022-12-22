#%%
from pathlib import Path
import pandas as pd

import julearn
from julearn.utils import logger, raise_error

from argparse import ArgumentParser

import sys

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

from lib.ml import get_data, run_cv  # noqa
from lib.constants import (
    FMRI_FEATURES,
    EEG_VISUAL_FEATURES,
    EEG_ABCD_FEATURES,
    EEG_MODEL_FEATURES,
    EEG_RESTING_FEATURES,
    TARGETS,
)

# %%

features = {
    "fmri": True,
    "eeg_resting": True,
    "eeg_visual": True,
    "eeg_abcd": True,
    "eeg_model": True,
}

df, _ = get_data(**features)
# %%

for t_target in TARGETS:
    print("===============")
    print(f"TARGET: {t_target}")
    print(df[t_target].value_counts())
    print("===============")

# %%
feature_sets = [
    {"features": {"eeg_visual": True}, "title": "VISUAL"},
    {
        "features": {"eeg_visual": True, "eeg_abcd": True},
        "title": "VISUAL+ABCD",
    },
    {
        "features": {
            "eeg_visual": True,
            "eeg_abcd": True,
            "eeg_resting": True,
        },
        "title": "VISUAL+ABCD+RESTING",
    },
    {
        "features": {"eeg_visual": True, "eeg_abcd": True, "eeg_model": True},
        "title": "VISUAL+ABCD+MODEL",
    },
    {"features": {"eeg_model": True}, "title": "MODEL"},
    {"features": {"fmri": True}, "title": "FMRI"},
    {"features": {"fmri": True, "eeg_model": True}, "title": "FMRI+MODEL"},
    {
        "features": {
            "fmri": True,
            "eeg_visual": True,
            "eeg_abcd": True,
            "eeg_model": True,
        },
        "title": "FMRI+VISUAL+ABCD+MODEL",
    },
    {
        "features": {
            "fmri": True,
            "eeg_resting": True,
            "eeg_visual": True,
            "eeg_abcd": True,
            "eeg_model": True,
        },
        "title": "FMRI+RESTING+VISUAL+ABCD+MODEL",
    },
    {"features": {"fmri": True, "eeg_visual": True}, "title": "FMRI+VISUAL"},
    {
        "features": {
            "eeg_resting": True,
            "eeg_visual": True,
            "eeg_abcd": True,
            "eeg_model": True,
        },
        "title": "RESTING+VISUAL+ABCD+MODEL",
    },
    {
        "features": {"fmri": True, "eeg_model": "resting"},
        "title": "FMRI+MODEL_RESTING",
    },
]

# %%
all_data_sizes = []
for t_set in feature_sets:
    features = t_set["features"]
    title = t_set["title"]
    t_df_data = []
    df, _ = get_data(**features)
    for t_target in TARGETS:
        t_df_data.append((df[t_target] == 1.0).value_counts())
    t_df_data = pd.concat(t_df_data, axis=1).transpose()
    t_df_data["set"] = title
    all_data_sizes.append(t_df_data)
# %%
all_data_sizes = pd.concat(all_data_sizes)

# %%
all_data_sizes.to_csv("./summary/data_sizes.csv", sep=";")
# %%
