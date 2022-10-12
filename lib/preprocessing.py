from copy import deepcopy
import numpy as np
from autoreject import AutoReject

import mne
from mne.utils import logger

from .constants import (
    _icm_lg_event_id,
    _icm_lg_concatenation_event,
    _egi256__copenhagen25_map,
    _egi256__copenhagen19_map,
    _copenhagen_ch_names,
)


def _check_min_events(epochs, min_events):
    n_orig_epochs = len([x for x in epochs.drop_log if "IGNORED" not in x])
    if isinstance(min_events, float):
        logger.info(
            "Using relative min_events: {} * {} = {} "
            "epochs remaining to reject preprocess".format(
                min_events, n_orig_epochs, int(n_orig_epochs * min_events)
            )
        )
        min_events = int(n_orig_epochs * min_events)

    epochs_remaining = len(epochs)
    if epochs_remaining < min_events:
        msg = (
            "Can not clean data. Only {} out of {} epochs "
            "remaining.".format(epochs_remaining, n_orig_epochs)
        )
        logger.error(msg)
        raise ValueError(msg)


def _ld_filter(raw, params=None, summary=None, n_jobs=1):
    if params is None:
        params = {}
    lpass = params.get("lpass", 45.0)
    hpass = params.get("hpass", 0.5)
    picks = mne.pick_types(
        raw.info, eeg=True, meg=False, ecg=False, exclude=[]
    )
    _filter_params = dict(
        method="iir",
        l_trans_bandwidth=0.1,
        iir_params=dict(ftype="butter", order=6),
    )
    filter_params = [
        dict(
            l_freq=hpass, h_freq=None, iir_params=dict(ftype="butter", order=6)
        ),
        dict(
            l_freq=None, h_freq=lpass, iir_params=dict(ftype="butter", order=8)
        ),
    ]

    for fp in filter_params:
        if fp["l_freq"] is None and fp["h_freq"] is None:
            continue
        _filter_params2 = deepcopy(_filter_params)
        if fp.get("method") == "fft":
            _filter_params2.pop("iir_params")
        if isinstance(fp, dict):
            _filter_params2.update(fp)
        if summary is not None:
            summary["steps"].append(
                {
                    "step": "filter",
                    "params": {
                        "hpass": "{} Hz".format(_filter_params2["l_freq"]),
                        "lpass": "{} Hz".format(_filter_params2["h_freq"]),
                    },
                }
            )
        raw.filter(picks=picks, n_jobs=n_jobs, **_filter_params2)

    notches = [50]
    if raw.info["sfreq"] > 200:
        notches.append(100)
    logger.info("Notch filters at {}".format(notches))
    if summary is not None:
        params = {(k + 1): "{} Hz".format(v) for k, v in enumerate(notches)}
        summary["steps"].append({"step": "notches", "params": params})
    if raw.info["sfreq"] != 250:
        logger.info("Resampling to 250Hz")
        raw = raw.resample(250)
        if summary is not None:
            summary["steps"].append(
                {"step": "resample", "params": {"sfreq": 250}}
            )
    raw.notch_filter(notches, method="fft", n_jobs=n_jobs)


def _egi_filter(raw, params=None, summary=None, n_jobs=1):
    if params is None:
        params = {}
    lpass = params.get("lpass", 45.0)
    hpass = params.get("hpass", 0.5)
    picks = mne.pick_types(raw.info, eeg=True, meg=True, ecg=True, exclude=[])
    _filter_params = dict(
        method="iir",
        l_trans_bandwidth=0.1,
        iir_params=dict(ftype="butter", order=6),
    )
    filter_params = [
        dict(
            l_freq=hpass, h_freq=None, iir_params=dict(ftype="butter", order=6)
        ),
        dict(
            l_freq=None, h_freq=lpass, iir_params=dict(ftype="butter", order=8)
        ),
    ]

    for fp in filter_params:
        if fp["l_freq"] is None and fp["h_freq"] is None:
            continue
        _filter_params2 = deepcopy(_filter_params)
        if fp.get("method") == "fft":
            _filter_params2.pop("iir_params")
        if isinstance(fp, dict):
            _filter_params2.update(fp)
        if summary is not None:
            summary["steps"].append(
                {
                    "step": "filter",
                    "params": {
                        "hpass": "{} Hz".format(_filter_params2["l_freq"]),
                        "lpass": "{} Hz".format(_filter_params2["h_freq"]),
                    },
                }
            )
        raw.filter(picks=picks, n_jobs=n_jobs, **_filter_params2)

    notches = [50, 100]
    logger.info("Notch filters at {}".format(notches))
    if summary is not None:
        params = {(k + 1): "{} Hz".format(v) for k, v in enumerate(notches)}
        summary["steps"].append({"step": "notches", "params": params})
    raw.notch_filter(notches, method="fft", n_jobs=n_jobs)

    if raw.info["sfreq"] != 250:
        logger.info("Resampling to 250Hz")
        raw.resample(250)
    if summary is not None:
        summary["steps"].append({"step": "resample", "params": {"sfreq": 250}})


def preprocess_rs_raw_ld(
    raw,
    t_cut=0.8,
    min_jitter=0.55,
    max_jitter=0.85,
    tmin=-0.2,
    baseline=None,
    min_events=50,
    n_jobs=1,
    do_summary=False,
):

    # Cut 800 ms epochs (-200, 600 ms)
    # between 550ms (550 + 800 = 1350 ms space between triggers)
    # and 850ms (850 + 800 = 1650 ms space between triggers)
    # reject = config_params.get('reject', None)

    # if reject is None:
    #     reject = {'eeg': 100e-6}

    summary = None
    if do_summary is True:
        summary = dict(steps=[], bad_channels=[])
    # Filter
    _ld_filter(raw, summary=summary, n_jobs=n_jobs)

    # Cut
    logger.info(f'Cutting events (sfreq = {raw.info["sfreq"]})')
    max_events = int(np.ceil(len(raw) / (raw.info["sfreq"] * t_cut))) + 1
    evt_times = []
    if isinstance(min_jitter, float):
        min_jitter = int(np.ceil(min_jitter * raw.info["sfreq"]))
    if isinstance(max_jitter, float):
        max_jitter = int(np.ceil(max_jitter * raw.info["sfreq"]))
    jitters = np.random.random_integers(min_jitter, max_jitter, max_events)
    epoch_len = int(np.ceil(t_cut * raw.info["sfreq"]))
    this_sample = 0
    this_jitter = 0
    while this_sample < len(raw):
        evt_times.append(this_sample + raw.first_samp)
        this_sample += epoch_len + jitters[this_jitter]
        this_jitter += 1
    evt_times = np.array(evt_times)
    events = np.concatenate(
        (
            evt_times[:, None],
            np.zeros((len(evt_times), 1), dtype=int),
            np.ones((len(evt_times), 1), dtype=int),
        ),
        axis=1,
    )
    event_id = 1

    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=tmin,
        tmax=t_cut + tmin,
        preload=True,
        reject=None,
        picks=None,
        baseline=baseline,
        verbose=False,
    )

    logger.info("Using autoreject")
    ar = AutoReject(n_interpolate=np.array([1, 2, 4, 8]))
    epochs_clean = ar.fit_transform(epochs)
    reject_log = ar.get_reject_log(epochs)
    if summary is not None:
        summary["autoreject"] = reject_log
        summary["steps"].append(
            dict(
                step="Autoreject",
                params={
                    "n_interpolate": ar.n_interpolate_["eeg"],
                    "consensus_perc": ar.consensus_["eeg"],
                },
                bad_epochs=np.where(reject_log.bad_epochs)[0],
            )
        )
    _check_min_events(epochs, min_events)
    logger.info(
        "found bad epochs: {} {}".format(
            np.sum(reject_log.bad_epochs), np.where(reject_log.bad_epochs)[0]
        )
    )
    epochs = epochs_clean

    ref_proj = mne.proj.make_eeg_average_ref_proj(epochs.info)
    epochs.add_proj(ref_proj)
    epochs.apply_proj()

    out = epochs
    if summary is not None:
        out = out, summary
    return out


def preprocess_lg_raw_egi(
    raw, tmin=-0.2, baseline=None, min_events=50, n_jobs=1, do_summary=False
):
    tmin = -0.2
    tmax = 1.34
    # run_ica = config_params.get('ica', False)
    baseline = (None, 0)

    summary = None
    if do_summary is True:
        summary = dict(steps=[], bad_channels=[])

    # Filter
    _egi_filter(raw, summary=summary, n_jobs=n_jobs)

    if raw.info["sfreq"] != 250:
        if summary is not None:
            summary["steps"].append(
                dict(step="Resample", params={"freq": 250, "npad": "auto"})
            )
        logger.info("Resampling to 250 Hz")
        raw.resample(250, npad="auto")
        logger.info("Resampling done")

    # Cut
    events = mne.find_events(raw, shortest_event=1)
    all_id = _icm_lg_event_id
    found_id = np.unique(events[:, 2])
    this_id = {k: v for k, v in all_id.items() if v in found_id}

    if summary is not None:
        summary["steps"].append(
            dict(step="Epochs", params={"baseline": baseline})
        )
    epochs = mne.Epochs(
        raw,
        events,
        this_id,
        tmin=tmin,
        tmax=tmax,
        preload=True,
        reject=None,
        picks=None,
        baseline=baseline,
        verbose=False,
    )

    ch_idx = epochs.ch_names.index("STI 014")
    concat_idx = []

    for ii, e in enumerate(epochs):
        if _icm_lg_concatenation_event in e[ch_idx]:  # constant
            concat_idx.append(ii)
    epochs.drop(concat_idx, reason="concatenation")
    epochs.drop_channels(["STI 014"])

    logger.info("Using autoreject")
    ar = AutoReject()
    epochs_clean = ar.fit_transform(epochs)
    reject_log = ar.get_reject_log(epochs)
    if summary is not None:
        summary["autoreject"] = reject_log
        summary["steps"].append(
            dict(
                step="Autoreject",
                params={
                    "n_interpolate": ar.n_interpolate_["eeg"],
                    "consensus_perc": ar.consensus_["eeg"],
                },
                bad_epochs=np.where(reject_log.bad_epochs)[0],
            )
        )
    _check_min_events(epochs, min_events)
    logger.info(
        "found bad epochs: {} {}".format(
            np.sum(reject_log.bad_epochs), np.where(reject_log.bad_epochs)[0]
        )
    )
    epochs = epochs_clean

    ref_proj = mne.proj.make_eeg_average_ref_proj(epochs.info)
    epochs.add_proj(ref_proj)
    epochs.apply_proj()

    out = epochs
    if summary is not None:
        out = out, summary
    return out


def map_montage(inst, n_channels):
    if n_channels == 25:
        rename = _egi256__copenhagen25_map
    elif n_channels == 19:
        rename = _egi256__copenhagen19_map
    else:
        raise ValueError(f"Wrong number of channels ({n_channels}) to map")
    to_keep = list(rename.keys())
    translated = inst.copy().pick_channels(to_keep)
    translated.rename_channels(rename)
    return translated.pick_channels(_copenhagen_ch_names[str(n_channels)])
