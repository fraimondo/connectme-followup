from pathlib import Path

import julearn
from julearn.utils import logger

import sys

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

from lib.ml import get_data, run_cv  # noqa
from lib.constants import (
    DIAG_FEATURES,
    FMRI_FEATURES,
    TARGETS,
    EEG_VISUAL_FEATURES,
    EEG_ABCD_FEATURES,
    EEG_MODEL_FEATURES,
    EEG_STIM_FEATURES,
    EEG_RESTING_FEATURES,
    AGESEX_FEATURES,
    DEATH_FEATURES,
    DIAGBIN_FEATURES,
    ALL_FEATURES
)

julearn.utils.configure_logging(level="INFO")


model = "svm"
cv = "kfold"

df, X = get_data(
    fmri=False,
    eeg_visual=True,
    eeg_abcd=True,
    eeg_model=True,
    eeg_stim=False,
    eeg_resting=False,
    diagnosis=False,
    drop_na=True,
    )

all_results = []
for y in TARGETS:
    title = "VISUAL+ABCD"
    result_df = run_cv(
        df, X, y, title=title, model=model, cv=cv, name="eeg_fmir_diag",
        target_name=y
    )
    all_results.append(result_df)
