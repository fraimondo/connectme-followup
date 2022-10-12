from warnings import warn
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectPercentile, f_classif


def get_clf_name(clf_type):
    name = None
    if clf_type == "gssvm":
        name = "GridSearch SVM"
    elif clf_type == "svm":
        name = "SVM"
    elif clf_type == "et-reduced":
        name = "ExtraTrees (reduced depth)"
    return name


def _get_cv(y, seed):
    n_splits = 5
    if y is not None:
        min_elems = min(np.sum(y), np.sum(1 - y))
        if min_elems < n_splits:
            warn(f"Using {min_elems - 1} splits instead of {n_splits}")
            n_splits = min_elems - 1
    else:
        warn("No labels provided, using 5 splits without checking")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return skf


def get_model(clf_type, seed, clf_params=None, clf_select=None, y=None):
    if clf_params is None:
        clf_params = {}
    if clf_type == "gssvm":
        skf = _get_cv(y, seed)
        cost_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        svc_params = dict(
            kernel="linear",
            probability=True,
            random_state=seed,
            class_weight="balanced",
        )
        svc_params.update(clf_params)
        print(f"Predict using GSSVM with {svc_params}")
        gc_fit_params = {"C": cost_range}
        print(f"GridSearch on {gc_fit_params}")
        GSSVM = GridSearchCV(
            SVC(**svc_params), gc_fit_params, cv=skf, scoring="roc_auc"
        )
        clf_model = GSSVM
        if clf_select is not None:
            print(f"Selecting {clf_select}% of markers")
            clf = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "select",
                        SelectPercentile(
                            score_func=f_classif, percentile=clf_select
                        ),
                    ),
                    (clf_type, clf_model),
                ]
            )
        else:
            clf = Pipeline(
                [("scaler", StandardScaler()), (clf_type, clf_model)]
            )
    elif clf_type == "svm":
        target_weight = np.sum(y == 1) / y.shape[0]
        class_weight = {0: 1 - target_weight, 1: target_weight}
        svc_params = dict(
            kernel="linear",
            probability=True,
            random_state=seed,
            class_weight=class_weight,
        )
        svc_params.update(clf_params)
        print(f"Predict using SVM with {svc_params}")
        clf_model = SVC(**svc_params)
        if clf_select is not None:
            print(f"Selecting {clf_select}% of markers")
            clf = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("select", SelectPercentile(f_classif, clf_select)),
                    (clf_type, clf_model),
                ]
            )
        else:
            clf = Pipeline(
                [("scaler", StandardScaler()), (clf_type, clf_model)]
            )
    elif clf_type == "et-reduced":
        et_params = dict(
            n_jobs=-1,
            n_estimators=2000,
            max_features=1,
            max_depth=4,
            random_state=seed,
            class_weight="balanced",
            criterion="entropy",
        )
        et_params.update(clf_params)
        print(f"Predict using ET (reduced depth) with {et_params}")
        clf_model = ExtraTreesClassifier(**et_params)
        clf = Pipeline([("scaler", RobustScaler()), (clf_type, clf_model)])
    else:
        raise ValueError("Unknown classifier: {}".format(clf_type))

    return clf
