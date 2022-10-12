from argparse import ArgumentParser

import pandas as pd
from pathlib import Path
import sys

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())
from lib.io import read_rs_ld_df  # noqa
from lib.preprocessing import preprocess_rs_raw_ld  # noqa
from lib.markers import get_resting_state  # noqa
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

fnames = list(db_path.glob(f'{subject}*.edf'))

raw = read_rs_ld_df(fnames)

epochs = preprocess_rs_raw_ld(raw)
n_channels = epochs.info['nchan']  # type: ignore

mc = get_resting_state()
mc.fit(epochs)


to_reduce_epochs = ['trim_mean80', 'std']
to_reduce_channels = ['mean', 'std']

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
final_df['session'] = 'stim' if 'stim' in fnames[0].name else 'resting'
final_df['channels'] = n_channels
out_path.mkdir(exist_ok=True, parents=True)
out_fname = out_path / f'{subject}_markers.csv'
final_df.to_csv(out_fname, sep=';')
