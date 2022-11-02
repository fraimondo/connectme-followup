#%% Imports
from pathlib import Path

import julearn
from julearn.utils import logger

import matplotlib.pyplot as plt
import seaborn as sns
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

#%% load data
df, X = get_data(
    fmri=False,
    eeg_visual=True,
    eeg_abcd=True,
    eeg_model=True,
    eeg_stim=True,
    eeg_resting=True,
    diagnosis=True,
    diagnosis_bin=True,
    drop_na=1,
    )
# %%
sns.swarmplot(x='GOS-E.3', y='doc.disch', data=df)

# %%
