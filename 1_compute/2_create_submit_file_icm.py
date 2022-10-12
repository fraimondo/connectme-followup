import os
from pathlib import Path


in_path = Path("/data/project/connectme/ICM/subjects")


out_path = "/data/project/connectme/ICM/results"

env = "nice"

cwd = os.getcwd()

log_dir = Path(cwd) / "logs" / "compute_baseline_icm"
log_dir.mkdir(exist_ok=True, parents=True)


exec_string = ("2_compute_baseline_icm.py --path $(dbpath) --opath $(opath) "
               "--subject $(subject)")

preamble = f"""
# The environment
universe       = vanilla
getenv         = True

# Resources
request_cpus   = 1
request_memory = 8G
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


submit_fname = "compute_baseline_icm.submit"


with open(submit_fname, "w") as submit_file:
    submit_file.write(preamble)

    subjects = [x.name for x in in_path.glob("*") if x.is_dir()]
    print(f'Found {len(subjects)} subjects')
    for t_subject in subjects:
        submit_file.write(f"subject={t_subject}\n")
        submit_file.write(f"dbpath={in_path.as_posix()}\n")
        submit_file.write(
            f"log_fname=compute_baseline_icm_{t_subject}\n"
        )
        submit_file.write(f"opath={out_path}\n")
        submit_file.write("queue\n\n")
