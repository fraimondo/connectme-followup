import os
from pathlib import Path


in_paths = ['/data/project/connectme/EEGs/edf/resting',
            '/data/project/connectme/EEGs/edf/stim']

in_paths = [Path(x) for x in in_paths]

env = 'nice'

cwd = os.getcwd()

log_dir = Path(cwd) / 'logs' / f'compute_markers'
log_dir.mkdir(exist_ok=True, parents=True)


exec_string = '1_compute_markers.py --path $(dbpath) --subject $(subject)'

preamble = f"""
# The environment
universe       = vanilla
getenv         = True

# Resources
request_cpus   = 1
request_memory = 2.5G
request_disk   = 0

# Executable
initial_dir    = {cwd}
executable     = {cwd}/run_in_venv.sh
transfer_executable = False

arguments      = {env} python {exec_string}

# Logs
log            = {log_dir.as_posix()}/$(log_fname).log
output         = {log_dir.as_posix()}/$(log_fname).out
error          = {log_dir.as_posix()}/$(log_fname).err

"""


submit_fname = 'compute_markers.submit'


with open(submit_fname, 'w') as submit_file:
    submit_file.write(preamble)

    for t_path in in_paths:
        subjects = [x.name for x in t_path.glob('*.edf')]
        subjects = [x.split('_')[0] for x in subjects]
        if 'stim' in t_path.name:
            kind = 'stim'
        else:
            kind = 'resting'
        for t_subject in subjects:
            submit_file.write(f'subject={t_subject}\n')
            submit_file.write(f'dbpath={t_path.as_posix()}\n')
            submit_file.write(
                f'log_fname=computer_markers_{t_subject}_{kind}\n')
            submit_file.write('queue\n\n')
