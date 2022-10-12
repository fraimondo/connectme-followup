from argparse import ArgumentParser

import pandas as pd
from pathlib import Path
import sys

import nice_ext.api as napi

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())
from lib.markers import get_resting_state  # noqa
from lib.preprocessing import map_montage  # noqa
from lib.reductions import get_reductions  # noqa


parser = ArgumentParser(description='Run the pipeline on the selected subject')

parser.add_argument('--path', metavar='path', nargs=1, type=str,
                    help='Path with the database.', required=True)

parser.add_argument('--opath', metavar='opath', nargs=1, type=str,
                    help='Path to store the results.', required=True)

parser.add_argument('--subject', metavar='subject', nargs=1, type=str,
                    help='Subject name', required=True)


args = parser.parse_args()
db_path = args.path
out_path = args.opath
subject = args.subject

if isinstance(db_path, list):
    db_path = db_path[0]

if isinstance(out_path, list):
    out_path = out_path[0]

if isinstance(subject, list):
    subject = subject[0]


db_path = Path(db_path)
out_path = Path(out_path)

s_path = db_path / subject

raw = napi.read(s_path, 'icm/lg/raw/egi')
epochs = napi.preprocess(raw, 'icm/lg/raw/egi')

n_channels = epochs.info['nchan']  # type: ignore

to_reduce_epochs = ['trim_mean80', 'std']
to_reduce_channels = ['mean', 'std']


for n_channels in [19, 25]:
    t_results = out_path / f'copenhagen_{n_channels}' / subject
    t_results.mkdir(parents=True, exist_ok=True)
    t_epochs = map_montage(epochs, n_channels)

    # Fit
    mc = get_resting_state()
    mc.fit(t_epochs)

    all_df = []
    for t_r_channels in to_reduce_channels:
        for t_r_epochs in to_reduce_epochs:
            t_reductions = get_reductions(
                n_channels, t_r_channels, t_r_epochs)
            red = mc.reduce_to_scalar(t_reductions)
            df = pd.DataFrame({'Marker': list(mc.keys()), 'Value': red})
            df['channels_fun'] = t_r_channels
            df['epochs_fun'] = t_r_epochs
            all_df.append(df)

    final_df = pd.concat(all_df, ignore_index=True)
    final_df['subject'] = subject
    out_fname = t_results / f'{subject}_markers.csv'
    final_df.to_csv(out_fname, sep=';')
