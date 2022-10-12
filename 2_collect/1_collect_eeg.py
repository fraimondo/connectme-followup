from pathlib import Path
import pandas as pd


in_paths = {
    "resting": "/data/project/connectme/EEGs/results/resting",
    "stim": "/data/project/connectme/EEGs/results/stim",
    "icm_19": "/data/project/connectme/ICM/results/copenhagen_19/",
    "icm_25": "/data/project/connectme/ICM/results/copenhagen_25/",
}

out_path = Path(__file__).parent.parent / "data"
out_path.mkdir(parents=True, exist_ok=True)

index_cols = [
    "subject",
    "Marker",
    "channels_fun",
    "epochs_fun",
]

opt_cols = ["session", "channels",]

stack_cols = ["Marker", "channels_fun", "epochs_fun"]

for t_name, t_path in in_paths.items():
    print(f"Processing {t_name}...")
    t_path = Path(t_path)
    all_dfs = []
    for t_file in t_path.glob("**/*.csv"):
        print(f"\t Collecting {t_file.name}")
        t_df = pd.read_csv(t_file, sep=";")
        all_dfs.append(t_df)

    final_df = pd.concat(all_dfs, ignore_index=True)

    t_index_cols = index_cols.copy()
    for x in opt_cols:
        if x in final_df.columns:
            t_index_cols.append(x)

    final_df = final_df.set_index(index_cols)["Value"].unstack(stack_cols)
    new_names = [
        f"{x.replace('nice/marker/', '').replace('/', '_')}_{y}_{z}"
        for x, y, z in final_df.columns]
    final_df.columns = new_names
    fname = out_path / f'{t_name}_EEG_markers.csv'
    print(f"Saving to {fname}")
    final_df.to_csv(fname, sep=';')
