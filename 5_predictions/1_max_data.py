from pathlib import Path
import pandas as pd

import julearn
from julearn.utils import logger, raise_error

from argparse import ArgumentParser

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

julearn.utils.configure_logging(level="DEBUG")

parser = ArgumentParser(description='Run the prediction with on max data')

parser.add_argument('--model', metavar='model', type=str,
                    help='Model to use.', required=True)
parser.add_argument('--cv', metavar='cv', type=str,
                    help='CV to use.', required=True)
parser.add_argument('--features', metavar='features', type=int,
                    help='features to use.', required=True)
parser.add_argument('--target', metavar='target', type=str,
                    help='target to use.', required=True)
parser.add_argument('--out-dir', metavar='out_dir', type=str,
                    help='out directory to use.', required=True)
args = parser.parse_args()
model = args.model
cv = args.cv
feature_set = args.features
target = args.target
out_dir = args.out_dir


title = "wrong"
features = {}

if feature_set == 1:
    features = {
        'eeg_visual': True,
    }
    title = "VISUAL"
elif feature_set == 2:
    features = {
        'eeg_visual': True,
        'eeg_abcd': True,
    }
    title = "VISUAL+ABCD"
elif feature_set == 3:
    features = {
        'eeg_visual': True,
        'eeg_abcd': True,
        'eeg_resting': True
    }
    title = "VISUAL+ABCD+RESTING"
elif feature_set == 4:
    features = {
        'eeg_visual': True,
        'eeg_abcd': True,
        'eeg_model': True
    }
    title = "VISUAL+ABCD+MODEL"
elif feature_set == 5:
    features = {
        'eeg_model': True
    }
    title = "MODEL"
elif feature_set == 6:
    features = {
        'fmri': True
    }
    title = "FMRI"
elif feature_set == 7:
    features = {
        'fmri': True,
        'eeg_model': True
    }
    title = "FMRI+MODEL"
elif feature_set == 8:
    features = {
        'fmri': True,
        'eeg_visual': True,
        'eeg_abcd': True,
        'eeg_model': True
    }
    title = "FMRI+VISUAL+ABCD+MODEL"
elif feature_set == 9:
    features = {
        'fmri': True,
        'eeg_resting': True,
        'eeg_visual': True,
        'eeg_abcd': True,
        'eeg_model': True
    }
    title = "FMRI+RESTING+VISUAL+ABCD+MODEL"
elif feature_set == 10:
    features = {
        'fmri': True,
        'eeg_visual': True
    }
    title = "FMRI+VISUAL"
elif feature_set == 11:
    features = {
        'eeg_resting': True,
        'eeg_visual': True,
        'eeg_abcd': True,
        'eeg_model': True
    }
    title = "RESTING+VISUAL+ABCD+MODEL"
elif feature_set == 12:
    features = {
        'fmri': True,
        'eeg_model': "resting",
    }
    title = "FMRI+MODEL_RESTING"
elif feature_set == 13:
    features = {
        'eeg_abcd': True,
    }
    title = "ABCD"
else:
    raise_error(f"Unknown feature set {feature_set}")

df, X = get_data(**features)

all_results = []
name = title.lower().replace('+', '_')
result_df = run_cv(
    df, X, target, title=title, model=model, cv=cv, name=name,
    target_name=target
)
all_results.append(result_df)

out_dir = Path(out_dir)
out_dir.mkdir(exist_ok=True, parents=True)
result_df = pd.concat(all_results)
fname = out_dir / f"set_{feature_set}_{model}_{target}_{cv}.csv"
result_df.to_csv(fname, sep=';')
