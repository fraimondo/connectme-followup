import pandas as pd
from pathlib import Path
import numpy as np

from sklearn.model_selection import cross_val_score

import sys

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

from lib.io import get_icm_scalars
from lib.ml import get_model

data_path = Path(__file__).parent.parent / "data"

n_channels = 19
kind = "resting"
pos_labels = ['MCS-', 'MCS+', 'EMCS']

train_data = get_icm_scalars(data_path, n_channels, pos_labels=pos_labels)

print("Train data:")
print(f"\tNumber of samples {train_data.shape[0]}")
print(f"\tNumber of features + target {train_data.shape[1]}")

X_train = train_data.drop(columns=["target"]).values
y_train = train_data["target"].values
print(f"\tNumber of features {X_train.shape[1]}")

clf = get_model(clf_type="gssvm", clf_select=90.0, seed=107, y=y_train)

# Small check
cv_scores = cross_val_score(
    clf, X_train, y_train, scoring="roc_auc", cv=10, n_jobs=-1
)
cv_score = np.mean(cv_scores)
print(f"CHECK CV Score = {cv_score}")

test_data = pd.read_csv(data_path / f"{kind}_EEG_markers.csv", sep=";")
print("Train data:")
print(f"\tNumber of samples {train_data.shape[0]}")
print(f"\tNumber of features + meta {test_data.shape[1]}")
X_test = train_data.drop(columns=["target"])