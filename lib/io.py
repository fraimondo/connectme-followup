from pathlib import Path
import mne
from mne.utils import logger

import sys
sys.path.append(Path(__file__).parent.parent.as_posix())

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
            '-AV', '').replace('LR', '').strip() for x in to_keep}

        t_raw.rename_channels(rename)

        # TODO: check between 19 or 25 electrodes
        str_chan = str(t_raw.info['nchan'])
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
    raws = sorted(raws, key=lambda x: x.info['meas_date'])
    raw = mne.io.concatenate_raws(raws)
    raw.set_montage(montage)
    raw.info['description'] = f'copenhagen/{str_chan}'
    logger.info('Reading done')
    return raw
