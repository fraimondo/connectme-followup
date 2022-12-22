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
features = {
    # 'fmri': True,
    'eeg_resting': True,
    'eeg_visual': True,
    'eeg_abcd': True,
    'eeg_model': True
}

df, _ = get_data(**features)

X = None

if feature_set == 1:
    X = EEG_VISUAL_FEATURES
    title = "VISUAL"
elif feature_set == 2:
    X = EEG_VISUAL_FEATURES + EEG_ABCD_FEATURES
    title = "VISUAL+ABCD"
elif feature_set == 3:
    X = EEG_VISUAL_FEATURES + EEG_ABCD_FEATURES + EEG_RESTING_FEATURES
    title = "VISUAL+ABCD+RESTING"
elif feature_set == 4:
    X = EEG_VISUAL_FEATURES + EEG_ABCD_FEATURES + EEG_MODEL_FEATURES
    title = "VISUAL+ABCD+MODEL"
elif feature_set == 5:
    X = EEG_MODEL_FEATURES
    title = "MODEL"
elif feature_set == 6:
    X = FMRI_FEATURES
    title = "FMRI"
    sys.exit("FMRI not implemented")
elif feature_set == 7:
    X = FMRI_FEATURES + EEG_MODEL_FEATURES
    title = "FMRI+MODEL"
    sys.exit("FMRI not implemented")
elif feature_set == 8:
    X = FMRI_FEATURES + EEG_VISUAL_FEATURES + \
        EEG_ABCD_FEATURES + EEG_MODEL_FEATURES
    title = "FMRI+VISUAL+ABCD+MODEL"
    sys.exit("FMRI not implemented")
elif feature_set == 9:
    X = FMRI_FEATURES + EEG_RESTING_FEATURES + EEG_VISUAL_FEATURES + \
        EEG_ABCD_FEATURES + EEG_MODEL_FEATURES
    title = "FMRI+RESTING+VISUAL+ABCD+MODEL"
    sys.exit("FMRI not implemented")
elif feature_set == 10:
    X = FMRI_FEATURES + EEG_VISUAL_FEATURES
    title = "FMRI+VISUAL"
    sys.exit("FMRI not implemented")
elif feature_set == 11:
    X = EEG_RESTING_FEATURES + EEG_VISUAL_FEATURES + \
        EEG_ABCD_FEATURES + EEG_MODEL_FEATURES
    title = "RESTING+VISUAL+ABCD+MODEL"
elif feature_set == 13:
    X = EEG_ABCD_FEATURES
    title = "ABCD"
else:
    raise_error(f"Unknown feature set {feature_set}")

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
