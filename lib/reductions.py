from tkinter import N
import numpy as np
from scipy.stats import trim_mean

from .constants import _copenhagen_19_rois, _copenhagen_25_rois


def entropy(a, axis=0):
    return -np.nansum(a * np.log(a), axis=axis) / np.log(a.shape[axis])


def trim_mean80(a, axis=0):
    return trim_mean(a, proportiontocut=.1, axis=axis)


def _get_func_by_name(name):
    if name == 'std':
        return np.std
    elif name == 'trim_mean80':
        return trim_mean80
    elif name == 'entropy':
        return entropy
    elif name == 'mean':
        return np.mean


def get_reductions(n_channels, channels_fun_name, epochs_fun_name):
    rois = _copenhagen_25_rois if n_channels == 25 else _copenhagen_19_rois
    scalp_roi = rois['scalp']
    epochs_fun = _get_func_by_name(epochs_fun_name)
    channels_fun = _get_func_by_name(channels_fun_name)
    return _get_reductions(scalp_roi, channels_fun, epochs_fun)


def _get_reductions(scalp_roi,channels_fun, epochs_fun):

    epochs_picks = None

    reduction_params = {}

    reduction_params['PowerSpectralDensity'] = {
        'reduction_func':
            [{'axis': 'frequency', 'function': np.sum},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': epochs_picks,
            'channels': scalp_roi}}

    reduction_params['PowerSpectralDensity/summary_se'] = {
        'reduction_func':
            [{'axis': 'frequency', 'function': entropy},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': epochs_picks,
            'channels': scalp_roi}}

    reduction_params['PowerSpectralDensitySummary'] = {
        'reduction_func':
            [{'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': epochs_picks,
            'channels': scalp_roi}}

    reduction_params['PermutationEntropy'] = {
        'reduction_func':
            [{'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': epochs_picks,
            'channels': scalp_roi}}

    reduction_params['SymbolicMutualInformation'] = {
        'reduction_func':
            [{'axis': 'channels_y', 'function': np.median},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': epochs_picks,
            'channels_y': scalp_roi,
            'channels': scalp_roi}}

    reduction_params['KolmogorovComplexity'] = {
        'reduction_func':
            [{'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': epochs_picks,
            'channels': scalp_roi}}

    return reduction_params
