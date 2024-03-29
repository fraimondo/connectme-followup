from warnings import warn
import numpy as np
from scipy import stats
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectPercentile, f_classif
import julearn
from julearn.pipeline import PipelineCreator

from julearn.utils import logger
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
    StratifiedKFold,
    LeaveOneOut,
)


from lib.constants import (
    DIAG_FEATURES,
    FMRI_FEATURES,
    TARGETS,
    EEG_VISUAL_FEATURES,
    EEG_ABCD_FEATURES,
    EEG_MODEL_FEATURES,
    EEG_MODEL_RESTING_FEATURES,
    EEG_STIM_FEATURES,
    EEG_RESTING_FEATURES,
    AGESEX_FEATURES,
    DEATH_FEATURES,
    DIAGBIN_FEATURES,
    CLINICAL_FEATURES,
    to_map,
)


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
        logger.info(f"Predict using GSSVM with {svc_params}")
        gc_fit_params = {"C": cost_range}
        logger.info(f"GridSearch on {gc_fit_params}")
        GSSVM = GridSearchCV(
            SVC(**svc_params), gc_fit_params, cv=skf, scoring="roc_auc"
        )
        clf_model = GSSVM
        if clf_select is not None:
            logger.info(f"Selecting {clf_select}% of markers")
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
        logger.info(f"Predict using SVM with {svc_params}")
        clf_model = SVC(**svc_params)
        if clf_select is not None:
            logger.info(f"Selecting {clf_select}% of markers")
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
        logger.info(f"Predict using ET (reduced depth) with {et_params}")
        clf_model = ExtraTreesClassifier(**et_params)
        clf = Pipeline([("scaler", RobustScaler()), (clf_type, clf_model)])
    else:
        raise ValueError("Unknown classifier: {}".format(clf_type))

    return clf


def get_data(
    fmri=False,
    eeg_visual=False,
    eeg_abcd=False,
    eeg_model=False,
    eeg_stim=False,
    eeg_resting=False,
    agesex=False,
    death=False,
    diagnosis=False,
    diagnosis_bin=False,
    clinical=False,
    drop_na=True,
):
    data_path = Path(__file__).parent.parent / "data" / "complete_df.2.csv"
    df = pd.read_csv(data_path, sep=";", decimal=",")

    X = []
    if fmri:
        X.extend(FMRI_FEATURES)
    if eeg_visual:
        X.extend(EEG_VISUAL_FEATURES)
    if eeg_abcd:
        X.extend(EEG_ABCD_FEATURES)

    if eeg_model is True:
        X.extend(EEG_MODEL_FEATURES)
    elif eeg_model == "resting":
        X.extend(EEG_MODEL_RESTING_FEATURES)

    if eeg_stim:
        X.extend(EEG_STIM_FEATURES)
    if eeg_resting:
        X.extend(EEG_RESTING_FEATURES)
    if agesex:
        X.extend(AGESEX_FEATURES)
    if death:
        X.extend(DEATH_FEATURES)
    if diagnosis:
        X.extend(DIAG_FEATURES)
    if diagnosis_bin:
        X.extend(DIAGBIN_FEATURES)
    if clinical:
        X.extend(CLINICAL_FEATURES)

    t_df = df[X + TARGETS]
    if drop_na is True or drop_na == 1:
        # Drop NANS
        t_df = t_df.dropna()

    if fmri is True:
        t_df[FMRI_FEATURES] = t_df[FMRI_FEATURES].astype(float)

    if eeg_model is True:
        t_df[EEG_MODEL_FEATURES] = t_df[EEG_MODEL_FEATURES].astype(float)
    elif eeg_model == "resting":
        t_df[EEG_MODEL_RESTING_FEATURES] = t_df[
            EEG_MODEL_RESTING_FEATURES
        ].astype(float)

    if eeg_stim is True:
        t_df[EEG_STIM_FEATURES] = t_df[EEG_STIM_FEATURES].astype(float)

    if eeg_resting is True:
        t_df[EEG_RESTING_FEATURES] = t_df[EEG_RESTING_FEATURES].astype(float)

    for t_col in t_df.columns:
        if t_col in to_map:
            this_map = to_map[t_col]
            t_df[t_col] = t_df[t_col].apply(lambda x: this_map[x])
            t_df[t_col] = t_df[t_col].astype(float)

    if drop_na is True or drop_na == 2:
        # Drop NAN again (in case mapping gives nan)
        t_df = t_df.dropna()

    return t_df, X


def proba_scorer(estimator, X, y):
    if X.shape[0] > 1:
        return 0
    return estimator.predict_proba(X)[:, 1]


def run_cv(df, X, y, title, model, cv, name, target_name):
    scoring = [
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
        "balanced_accuracy",
    ]
    if cv == "kfold":
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    elif cv == "mc":
        cv = StratifiedShuffleSplit(
            n_splits=100, test_size=0.3, random_state=42
        )
    elif cv == "loo":
        cv = LeaveOneOut()
        julearn.scoring.register_scorer("proba", proba_scorer)
        scoring = ["accuracy", "proba"]
    else:
        raise ValueError('Unknown CV scheme ["kfold" or "mc"]')

    search_params = None
    creator = PipelineCreator(problem_type="classification")
    if y in ["GOS-E.3", "GOS-E.12"]:
        creator = PipelineCreator(problem_type="regression")
        scoring = ["r2", "neg_mean_absolute_error", "neg_mean_squared_error"]

    creator.add("zscore")
    if model == "svm":
        creator.add(
            "svm", kernel="linear", class_weight="balanced", probability=True
        )

    elif model == "rf":
        creator.add("rf", n_estimators=500)
    elif model == "gssvm":
        if y in ["GOS-E.3", "GOS-E.12"]:
            creator.add(
                "svm",
                kernel="linear",
                C=[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2],
                class_weight="balanced",
                probability=True,
                epsilon=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            )
        else:
            creator.add(
                "svm",
                kernel="linear",
                C=[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2],
                class_weight="balanced",
                probability=True,
            )

        search_params = {
            "cv": StratifiedKFold(n_splits=5, random_state=77, shuffle=True),
        }
    elif model == "gsrf":
        search_params = {
            "cv": StratifiedKFold(n_splits=5, random_state=77, shuffle=True),
        }
        max_depths = [1, int(len(X) / 4), int(len(X) / 2), len(X)]
        max_depths = [x for x in max_depths if x > 0]
        creator.add(
            "rf",
            n_estimators=[200, 500],
            max_depth=max_depths,
            max_features=["sqrt", "log2"],
            criterion=["gini", "entropy", "log_loss"],
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unknown model {model}")

    cv_results, final_model, inspector = julearn.api.run_cross_validation(
        X=X,
        y=y,
        data=df,
        X_types={"continuous": X},
        model=creator,
        scoring=scoring,
        cv=cv,
        return_estimator="all",
        return_inspector=True,
        return_train_score=True,
    )
    cv_results["features"] = name  # type: ignore
    cv_results["model"] = model  # type: ignore
    cv_results["target"] = y  # type: ignore
    logger.info("=============================")
    logger.info(title)
    logger.info(f"TARGET: {target_name}")
    logger.info(f"# SAMPLES: {len(df)}")
    logger.info(cv_results.mean(numeric_only=True))  # type: ignore
    logger.info("=============================")

    return cv_results, final_model, inspector


def compute_ci(data, ci=95, use_percentile=False, use_gaussian=False):
    x = np.mean(data)

    if use_percentile:
        ci_lower, ci_upper = np.percentile(
            data, [(100 - ci) / 2, ci + ((100 - ci) / 2)]
        )
    elif use_gaussian:
        if ci != 95:
            raise ValueError("Gaussian CI only supports 95%")
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1
        ci_lower = med - 1.57 * iqr / np.sqrt(len(data))
        ci_upper = med + 1.57 * iqr / np.sqrt(len(data))
    else:
        n_samples = len(data)
        ci = ci / 100
        if n_samples <= 30:
            ci_lower, ci_upper = stats.t.interval(
                ci, len(data) - 1, loc=x, scale=stats.sem(data)
            )
        else:
            ci_lower, ci_upper = stats.norm.interval(
                alpha=ci, loc=np.mean(data), scale=stats.sem(data)
            )
    # return pd.Series({"mean": x, "ci_lower": ci_lower, "ci_upper": ci_upper})
    return x, ci_lower, ci_upper
