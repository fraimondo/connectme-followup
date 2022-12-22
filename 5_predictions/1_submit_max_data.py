from pathlib import Path
import os

# feature_sets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# feature_sets = [6, 10, 12]
feature_sets = [13]
targets = [
    "GOS-E.3.bin",
    "GOS-E.12.bin",
    "mRS.3.bin",
    "mRS.12.bin",
    "CPC.3.bin",
    "CPC.12.bin",
    "combined.outcome.3",
    "combined.outcome.12",
]
# models = ["gsrf"]
models = ["gssvm", "gsrf", "rf", "svm"]
cvs = ["kfold"]
# cvs = ["loo"]

out_dir = Path(__file__).resolve().parent / 'results' / '1_max_data'
out_dir.mkdir(exist_ok=True, parents=True)

env = "julearn"

cwd = os.getcwd()

log_dir = Path(cwd) / "logs" / "1_max_data"
log_dir.mkdir(exist_ok=True, parents=True)


exec_string = ("1_max_data.py --features $(features) --cv $(cv) "
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


submit_fname = "1_max_data.submit"

with open(submit_fname, "w") as submit_file:
    submit_file.write(preamble)
    for cv in cvs:
        (log_dir / cv).mkdir(exist_ok=True, parents=True)
        for feature_set in feature_sets:
            for model in models:
                for target in targets:
                    submit_file.write(f"model={model}\n")
                    submit_file.write(f"cv={cv}\n")
                    submit_file.write(f"target={target}\n")
                    submit_file.write(f"features={feature_set}\n")
                    submit_file.write(
                        f"log_fname={cv}/1_max_data_set_"
                        f"{feature_set}_{model}_{target}_{cv}\n"
                    )
                    submit_file.write("queue\n\n")
