from pathlib import Path
import pandas as pd
import sys

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

from lib.io import get_connectme_scalars

data_path = Path(__file__).parent.parent / "data"

df_meta = pd.read_csv(
    data_path / "dataset.Federico.followup.final.csv", sep=";"
)

# Filter NAN subjects
df_meta = df_meta[~df_meta["Id"].isna()]

# Rename some columns
columns_to_keep = [
    "Id",
    "age",
    "sex",
    "sed.exam.enrolment",
    "sed.eeg",
    "sed.fmri",
    "eeg.visual.synek",
    "abcd",
    "dmn.dmn",
    "fpn.fpn",
    "sn.sn",
    "an.an",
    "smn.smn",
    "vn.vn",
    "dmn.fpn",
    "dmn.sn",
    "dmn.an",
    "dmn.smn",
    "dmn.vn",
    "fpn.sn",
    "fpn.an",
    "fpn.smn",
    "fpn.vn",
    "sn.an",
    "sn.smn",
    "sn.vn",
    "an.smn",
    "an.vn",
    "smn.vn",
    "doc.enrol",
    "doc.enrol.bi",
    "doc.disch",
    "doc.disch.bi",
    "Death at ICU discharge",
    "cause.death (none=0, WLST=1, Other=2)",
    "GOS-E.3",
    "GOS-E.3.bin",
    "GOS-E.12",
    "GOS-E.12.bin",
    "mRS.3.bin",
    "mRS.12.bin",
    "CPC.3.bin",
    "CPC.12.bin",
]

df_meta = df_meta[columns_to_keep]

rename_dict = {
    "Id": "subject",
    "abcd": "eeg.abcd",
    "Death at ICU discharge": "death",
    "cause.death (none=0, WLST=1, Other=2)": "death.cause",
}

fmri_vars = [
    "dmn.dmn",
    "fpn.fpn",
    "sn.sn",
    "an.an",
    "smn.smn",
    "vn.vn",
    "dmn.fpn",
    "dmn.sn",
    "dmn.an",
    "dmn.smn",
    "dmn.vn",
    "fpn.sn",
    "fpn.an",
    "fpn.smn",
    "fpn.vn",
    "sn.an",
    "sn.smn",
    "sn.vn",
    "an.smn",
    "an.vn",
    "smn.vn",
]

for t_var in fmri_vars:
    rename_dict[t_var] = f"fmri.{t_var}"
df_meta.rename(columns=rename_dict, inplace=True)

# Add EEG markers
eeg_dfs = []
for n_channels in [19, 25]:
    c_df = None
    for kind in ["resting", "stim"]:
        t_data = get_connectme_scalars(data_path, n_channels, kind=kind)
        to_rename = {x: f"eeg.{kind}.{x}" for x in t_data.columns}
        t_data.rename(columns=to_rename, inplace=True)
        if c_df is None:
            c_df = t_data
        else:
            c_df = c_df.merge(t_data, on="subject", how="outer")
    c_df["eeg.n_channels"] = n_channels
    eeg_dfs.append(c_df)

eeg_df = pd.concat(eeg_dfs, axis=0)


df_meta = df_meta.merge(eeg_df, on="subject", how="outer")

# Add ICM predictions for diagnosis
icm_df = pd.read_csv(data_path / "icm_predictions.csv", sep=";")

icm_df.rename(
    columns={
        "PMCS_resting": 'eeg.stack.PMCS_resting',
        "PMCS_stim": 'eeg.stack.PMCS_stim',
    },
    inplace=True,
)
icm_df = icm_df.drop(columns=['channels'])

df_meta = df_meta.merge(icm_df, on="subject", how="outer")

df_meta.to_csv(data_path / "complete_df.csv", sep=";", index=False)