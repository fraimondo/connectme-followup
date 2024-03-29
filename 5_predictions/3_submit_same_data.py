from pathlib import Path
import os

# feature_sets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# # feature_sets = [1, 2, 3, 4, 5, 11]
feature_sets = [16]
# feature_sets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
targets = [
    # "GOS-E.3.bin",
    # "GOS-E.12.bin",
    # "mRS.3.bin",
    # "mRS.12.bin",
    # "CPC.3.bin",
    # "CPC.12.bin",
    "combined.outcome.3",
    "combined.outcome.12",
]
models = ["gssvm", "gsrf"]
cvs = ["kfold"]

out_dir = Path(__file__).resolve().parent / 'results' / '3_same_data'
out_dir.mkdir(exist_ok=True, parents=True)

env = "julearn"

cwd = os.getcwd()

log_dir = Path(cwd) / "logs" / "3_same_data"
log_dir.mkdir(exist_ok=True, parents=True)


exec_string = ("3_same_data.py --features $(features) --cv $(cv) "
               "--model $(model) --target $(target) "
               f"--out-dir {out_dir.as_posix()}")

preamble = f"""
# The environment
universe       = vanilla
getenv         = True

# Resources
request_cpus   = 1
request_memory = 16G
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


submit_fname = "3_same_data.submit"

with open(submit_fname, "w") as submit_file:
    submit_file.write(preamble)
    for cv in cvs:
        for feature_set in feature_sets:
            for model in models:
                for target in targets:
                    submit_file.write(f"model={model}\n")
                    submit_file.write(f"cv={cv}\n")
                    submit_file.write(f"target={target}\n")
                    submit_file.write(f"features={feature_set}\n")
                    submit_file.write(
                        "log_fname=3_same_data_set"
                        f"{feature_set}_{model}_{target}_{cv}\n"
                    )
                    submit_file.write("queue\n\n")
