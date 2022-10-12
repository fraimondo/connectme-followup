import numpy as np
from pathlib import Path
import mne
from mne.utils import logger

from lib.constants import _copenhagen_montages, _copenhagen_ch_names


def read_rs_ld_df(fnames):
    raws = []
    for t_fname in fnames:
        if not isinstance(t_fname, Path):
            t_fname = Path(t_fname)
        logger.info(f'Reading from {t_fname.as_posix()}')
        if t_fname.name.endswith('edf'):
            t_raw = mne.io.read_raw_edf(t_fname, preload=True, verbose=True)
        else:
            t_raw = mne.io.read_raw_fif(t_fname, preload=True, verbose=True)

        to_keep = [x for x in t_raw.ch_names if x.startswith('EEG')]
        to_drop = [x for x in t_raw.ch_names if x not in to_keep]
        t_raw.drop_channels(to_drop)

        rename = {x: x.replace('EEG', '').replace('-REF', '').replace(
            '-AV', '').replace('LR', '').replace('-Ref', '').strip()
            for x in to_keep}

        t_raw.rename_channels(rename)

        # TODO: check between 19 or 25 electrodes
        n_channels = t_raw.info['nchan']
        if n_channels < 19:
            raise ValueError('Less than 19 channels in data. Check!')
        elif n_channels < 25:
            n_channels = 19
        else:
            n_channels = 25
        str_chan = str(n_channels)
        montage = mne.channels.make_standard_montage(
            _copenhagen_montages[str_chan])
        chs_to_keep = _copenhagen_ch_names[str_chan]

        t_raw.pick_channels(chs_to_keep)

        meas_date = t_raw.info['meas_date']
        orig_time = t_raw.annotations.orig_time
        if orig_time != meas_date:
            raise ValueError('Annotations are not aligned to recording')

        logger.info('Adding standard channel locations to info.')
        t_raw.set_montage(montage)
        t_raw.info['description'] = f'copenhagen/{str_chan}'
        raws.append(t_raw)

    all_n_channels = [t.info['nchan'] for t in raws]
    if len(np.unique(all_n_channels)) > 1:
        n_to_keep = np.min(all_n_channels)
        str_chan = str(n_to_keep)
        chs_to_keep = _copenhagen_ch_names[str_chan]
        for t in raws:
            t.pick_channels(chs_to_keep)
    raws = sorted(raws, key=lambda x: x.info['meas_date'])
    raw = mne.io.concatenate_raws(raws)
    raw.set_montage(montage)
    raw.info['description'] = f'copenhagen/{str_chan}'
    logger.info('Reading done')
    return raw

