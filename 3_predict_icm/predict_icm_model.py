import pandas as pd
from pathlib import Path
import numpy as np

from sklearn.model_selection import cross_val_score

import sys

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

from lib.io import get_icm_scalars, get_connectme_scalars
from lib.ml import get_model

data_path = Path(__file__).parent.parent / "data"

do_check = True
pos_labels = ["MCS-", "MCS+", "EMCS"]


all_df = []
for n_channels in [19, 25]:
    print("================")
    print(f"Channels: {n_channels}")
    print("================")
    train_data = get_icm_scalars(data_path, n_channels, pos_labels=pos_labels)

    print("Train data:")
    print(f"\tNumber of samples {train_data.shape[0]}")
    print(f"\tNumber of features + target {train_data.shape[1]}")

    X_train = train_data.drop(columns=["target"]).values
    y_train = train_data["target"].values
    print(f"\tNumber of features {X_train.shape[1]}")

    clf = get_model(clf_type="gssvm", clf_select=90.0, seed=107, y=y_train)

    if do_check is True:
        # Small check
        cv_scores = cross_val_score(
            clf, X_train, y_train, scoring="roc_auc", cv=10, n_jobs=-1
        )
        cv_score = np.mean(cv_scores)
        print(f"CHECK CV Score = {cv_score}")

    clf.fit(X_train, y_train)

    channels_df = None
    for kind in ["resting", "stim"]:

        test_data = get_connectme_scalars(data_path, n_channels, kind=kind)
        X_test = test_data.values
        print("Test data:")
        print(f"\tNumber of samples {X_test.shape[0]}")
        print(f"\tNumber of features {X_test.shape[1]}")
        probas = clf.predict_proba(X_test)[:, 1]

        t_df = pd.DataFrame(
            index=test_data.index, data=probas, columns=[f"PMCS_{kind}"]
        )
        # t_df['channels'] = n_channels
        if channels_df is None:
            channels_df = t_df
        else:
            channels_df = channels_df.join(t_df)

    channels_df['channels'] = n_channels
    all_df.append(channels_df)
print("Saving results")
final_df = pd.concat(all_df)
final_df.to_csv(data_path / "icm_predictions.csv", sep=";")