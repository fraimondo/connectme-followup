import numpy as np


_copenhagen_montages = {"25": "standard_1020", "19": "standard_1020"}

_copenhagen_ch_names = {
    "25": [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T7",
        "T8",
        "P7",
        "P8",
        "T9",
        "T10",
        "Fz",
        "Cz",
        "Pz",
        "F10",
        "F9",
        "P9",
        "P10",
    ],
    "19": [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T7",
        "T8",
        "P7",
        "P8",
        "Fz",
        "Cz",
        "Pz",
    ],
}


_copenhagen_25_rois = {
    "p3a": np.array([2, 3, 18, 19, 4, 5]),  # Fz + Cz
    "p3b": np.array([19, 4, 5, 20, 6, 7]),  # Cz + Pz
    "mmn": np.array([2, 3, 18, 19, 4, 5]),  # Fz + Cz
    "cnv": np.array([2, 3, 18, 19, 4, 5]),  # Fz + Cz
    "Fz": np.array([2, 3, 18]),
    "Cz": np.array([19, 4, 5]),
    "Pz": np.array([20, 6, 7]),
    "scalp": np.array(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
        ]
    ),
    "nonscalp": None,
}

_copenhagen_19_rois = {
    "p3a": np.array([2, 3, 16, 17, 4, 5]),  # Fz + Cz
    "p3b": np.array([17, 4, 5, 18, 6, 7]),  # Cz + Pz
    "mmn": np.array([2, 3, 16, 17, 4, 5]),  # Fz + Cz
    "cnv": np.array([2, 3, 16, 17, 4, 5]),  # Fz + Cz
    "Fz": np.array([2, 3, 16]),
    "Cz": np.array([17, 4, 5]),
    "Pz": np.array([18, 6, 7]),
    "scalp": np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    ),
    "nonscalp": None,
}


_egi256__copenhagen25_map = {
    "E37": "Fp1",
    "E18": "Fp2",
    "E36": "F3",
    "E224": "F4",
    "E59": "C3",
    "E183": "C4",
    "E87": "P3",
    "E153": "P4",
    "E116": "O1",
    "E150": "O2",
    "E47": "F7",
    "E2": "F8",
    "E69": "T7",
    "E202": "T8",
    "E96": "P7",
    "E170": "P8",
    "E68": "T9",
    "E210": "T10",
    "E21": "Fz",
    "E81": "Cz",
    "E101": "Pz",
    "E226": "F10",
    "E252": "F9",
    "E106": "P9",
    "E169": "P10",
}

_egi256__copenhagen19_map = {
    "E37": "Fp1",
    "E18": "Fp2",
    "E36": "F3",
    "E224": "F4",
    "E59": "C3",
    "E183": "C4",
    "E87": "P3",
    "E153": "P4",
    "E116": "O1",
    "E150": "O2",
    "E47": "F7",
    "E2": "F8",
    "E69": "T7",
    "E202": "T8",
    "E96": "P7",
    "E170": "P8",
    "E21": "Fz",
    "E81": "Cz",
    "E101": "Pz",
}

_icm_lg_event_id = {
    "HSTD": 10,
    "HDVT": 20,
    "LSGS": 30,
    "LSGD": 40,
    "LDGS": 60,
    "LDGD": 50,
}

_icm_lg_concatenation_event = 2014.0

# FEATURE SETS

FMRI_FEATURES = [
    "fmri.dmn.dmn",
    "fmri.fpn.fpn",
    "fmri.sn.sn",
    "fmri.an.an",
    "fmri.smn.smn",
    "fmri.vn.vn",
    "fmri.dmn.fpn",
    "fmri.dmn.sn",
    "fmri.dmn.an",
    "fmri.dmn.smn",
    "fmri.dmn.vn",
    "fmri.fpn.sn",
    "fmri.fpn.an",
    "fmri.fpn.smn",
    "fmri.fpn.vn",
    "fmri.sn.an",
    "fmri.sn.smn",
    "fmri.sn.vn",
    "fmri.an.smn",
    "fmri.an.vn",
    "fmri.smn.vn",
]

EEG_VISUAL_FEATURES = ["eeg.visual.synek"]
EEG_ABCD_FEATURES = ["eeg.abcd"]
EEG_MODEL_FEATURES = ["eeg.stack.PMCS_resting", "eeg.stack.PMCS_stim"]
EEG_STIM_FEATURES = [
    "eeg.stim.PowerSpectralDensity_delta_mean_trim_mean80",
    "eeg.stim.PowerSpectralDensity_deltan_mean_trim_mean80",
    "eeg.stim.PowerSpectralDensity_theta_mean_trim_mean80",
    "eeg.stim.PowerSpectralDensity_thetan_mean_trim_mean80",
    "eeg.stim.PowerSpectralDensity_alpha_mean_trim_mean80",
    "eeg.stim.PowerSpectralDensity_alphan_mean_trim_mean80",
    "eeg.stim.PowerSpectralDensity_beta_mean_trim_mean80",
    "eeg.stim.PowerSpectralDensity_betan_mean_trim_mean80",
    "eeg.stim.PowerSpectralDensity_gamma_mean_trim_mean80",
    "eeg.stim.PowerSpectralDensity_gamman_mean_trim_mean80",
    "eeg.stim.PowerSpectralDensity_summary_se_mean_trim_mean80",
    "eeg.stim.PowerSpectralDensitySummary_summary_msf_mean_trim_mean80",
    "eeg.stim.PowerSpectralDensitySummary_summary_sef90_mean_trim_mean80",
    "eeg.stim.PowerSpectralDensitySummary_summary_sef95_mean_trim_mean80",
    "eeg.stim.PermutationEntropy_default_mean_trim_mean80",
    "eeg.stim.SymbolicMutualInformation_weighted_mean_trim_mean80",
    "eeg.stim.KolmogorovComplexity_default_mean_trim_mean80",
    "eeg.stim.PowerSpectralDensity_delta_mean_std",
    "eeg.stim.PowerSpectralDensity_deltan_mean_std",
    "eeg.stim.PowerSpectralDensity_theta_mean_std",
    "eeg.stim.PowerSpectralDensity_thetan_mean_std",
    "eeg.stim.PowerSpectralDensity_alpha_mean_std",
    "eeg.stim.PowerSpectralDensity_alphan_mean_std",
    "eeg.stim.PowerSpectralDensity_beta_mean_std",
    "eeg.stim.PowerSpectralDensity_betan_mean_std",
    "eeg.stim.PowerSpectralDensity_gamma_mean_std",
    "eeg.stim.PowerSpectralDensity_gamman_mean_std",
    "eeg.stim.PowerSpectralDensity_summary_se_mean_std",
    "eeg.stim.PowerSpectralDensitySummary_summary_msf_mean_std",
    "eeg.stim.PowerSpectralDensitySummary_summary_sef90_mean_std",
    "eeg.stim.PowerSpectralDensitySummary_summary_sef95_mean_std",
    "eeg.stim.PermutationEntropy_default_mean_std",
    "eeg.stim.SymbolicMutualInformation_weighted_mean_std",
    "eeg.stim.KolmogorovComplexity_default_mean_std",
    "eeg.stim.PowerSpectralDensity_delta_std_trim_mean80",
    "eeg.stim.PowerSpectralDensity_deltan_std_trim_mean80",
    "eeg.stim.PowerSpectralDensity_theta_std_trim_mean80",
    "eeg.stim.PowerSpectralDensity_thetan_std_trim_mean80",
    "eeg.stim.PowerSpectralDensity_alpha_std_trim_mean80",
    "eeg.stim.PowerSpectralDensity_alphan_std_trim_mean80",
    "eeg.stim.PowerSpectralDensity_beta_std_trim_mean80",
    "eeg.stim.PowerSpectralDensity_betan_std_trim_mean80",
    "eeg.stim.PowerSpectralDensity_gamma_std_trim_mean80",
    "eeg.stim.PowerSpectralDensity_gamman_std_trim_mean80",
    "eeg.stim.PowerSpectralDensity_summary_se_std_trim_mean80",
    "eeg.stim.PowerSpectralDensitySummary_summary_msf_std_trim_mean80",
    "eeg.stim.PowerSpectralDensitySummary_summary_sef90_std_trim_mean80",
    "eeg.stim.PowerSpectralDensitySummary_summary_sef95_std_trim_mean80",
    "eeg.stim.PermutationEntropy_default_std_trim_mean80",
    "eeg.stim.SymbolicMutualInformation_weighted_std_trim_mean80",
    "eeg.stim.KolmogorovComplexity_default_std_trim_mean80",
    "eeg.stim.PowerSpectralDensity_delta_std_std",
    "eeg.stim.PowerSpectralDensity_deltan_std_std",
    "eeg.stim.PowerSpectralDensity_theta_std_std",
    "eeg.stim.PowerSpectralDensity_thetan_std_std",
    "eeg.stim.PowerSpectralDensity_alpha_std_std",
    "eeg.stim.PowerSpectralDensity_alphan_std_std",
    "eeg.stim.PowerSpectralDensity_beta_std_std",
    "eeg.stim.PowerSpectralDensity_betan_std_std",
    "eeg.stim.PowerSpectralDensity_gamma_std_std",
    "eeg.stim.PowerSpectralDensity_gamman_std_std",
    "eeg.stim.PowerSpectralDensity_summary_se_std_std",
    "eeg.stim.PowerSpectralDensitySummary_summary_msf_std_std",
    "eeg.stim.PowerSpectralDensitySummary_summary_sef90_std_std",
    "eeg.stim.PowerSpectralDensitySummary_summary_sef95_std_std",
    "eeg.stim.PermutationEntropy_default_std_std",
    "eeg.stim.SymbolicMutualInformation_weighted_std_std",
    "eeg.stim.KolmogorovComplexity_default_std_std",
]
EEG_RESTING_FEATURES = [
    "eeg.resting.PowerSpectralDensity_delta_mean_trim_mean80",
    "eeg.resting.PowerSpectralDensity_deltan_mean_trim_mean80",
    "eeg.resting.PowerSpectralDensity_theta_mean_trim_mean80",
    "eeg.resting.PowerSpectralDensity_thetan_mean_trim_mean80",
    "eeg.resting.PowerSpectralDensity_alpha_mean_trim_mean80",
    "eeg.resting.PowerSpectralDensity_alphan_mean_trim_mean80",
    "eeg.resting.PowerSpectralDensity_beta_mean_trim_mean80",
    "eeg.resting.PowerSpectralDensity_betan_mean_trim_mean80",
    "eeg.resting.PowerSpectralDensity_gamma_mean_trim_mean80",
    "eeg.resting.PowerSpectralDensity_gamman_mean_trim_mean80",
    "eeg.resting.PowerSpectralDensity_summary_se_mean_trim_mean80",
    "eeg.resting.PowerSpectralDensitySummary_summary_msf_mean_trim_mean80",
    "eeg.resting.PowerSpectralDensitySummary_summary_sef90_mean_trim_mean80",
    "eeg.resting.PowerSpectralDensitySummary_summary_sef95_mean_trim_mean80",
    "eeg.resting.PermutationEntropy_default_mean_trim_mean80",
    "eeg.resting.SymbolicMutualInformation_weighted_mean_trim_mean80",
    "eeg.resting.KolmogorovComplexity_default_mean_trim_mean80",
    "eeg.resting.PowerSpectralDensity_delta_mean_std",
    "eeg.resting.PowerSpectralDensity_deltan_mean_std",
    "eeg.resting.PowerSpectralDensity_theta_mean_std",
    "eeg.resting.PowerSpectralDensity_thetan_mean_std",
    "eeg.resting.PowerSpectralDensity_alpha_mean_std",
    "eeg.resting.PowerSpectralDensity_alphan_mean_std",
    "eeg.resting.PowerSpectralDensity_beta_mean_std",
    "eeg.resting.PowerSpectralDensity_betan_mean_std",
    "eeg.resting.PowerSpectralDensity_gamma_mean_std",
    "eeg.resting.PowerSpectralDensity_gamman_mean_std",
    "eeg.resting.PowerSpectralDensity_summary_se_mean_std",
    "eeg.resting.PowerSpectralDensitySummary_summary_msf_mean_std",
    "eeg.resting.PowerSpectralDensitySummary_summary_sef90_mean_std",
    "eeg.resting.PowerSpectralDensitySummary_summary_sef95_mean_std",
    "eeg.resting.PermutationEntropy_default_mean_std",
    "eeg.resting.SymbolicMutualInformation_weighted_mean_std",
    "eeg.resting.KolmogorovComplexity_default_mean_std",
    "eeg.resting.PowerSpectralDensity_delta_std_trim_mean80",
    "eeg.resting.PowerSpectralDensity_deltan_std_trim_mean80",
    "eeg.resting.PowerSpectralDensity_theta_std_trim_mean80",
    "eeg.resting.PowerSpectralDensity_thetan_std_trim_mean80",
    "eeg.resting.PowerSpectralDensity_alpha_std_trim_mean80",
    "eeg.resting.PowerSpectralDensity_alphan_std_trim_mean80",
    "eeg.resting.PowerSpectralDensity_beta_std_trim_mean80",
    "eeg.resting.PowerSpectralDensity_betan_std_trim_mean80",
    "eeg.resting.PowerSpectralDensity_gamma_std_trim_mean80",
    "eeg.resting.PowerSpectralDensity_gamman_std_trim_mean80",
    "eeg.resting.PowerSpectralDensity_summary_se_std_trim_mean80",
    "eeg.resting.PowerSpectralDensitySummary_summary_msf_std_trim_mean80",
    "eeg.resting.PowerSpectralDensitySummary_summary_sef90_std_trim_mean80",
    "eeg.resting.PowerSpectralDensitySummary_summary_sef95_std_trim_mean80",
    "eeg.resting.PermutationEntropy_default_std_trim_mean80",
    "eeg.resting.SymbolicMutualInformation_weighted_std_trim_mean80",
    "eeg.resting.KolmogorovComplexity_default_std_trim_mean80",
    "eeg.resting.PowerSpectralDensity_delta_std_std",
    "eeg.resting.PowerSpectralDensity_deltan_std_std",
    "eeg.resting.PowerSpectralDensity_theta_std_std",
    "eeg.resting.PowerSpectralDensity_thetan_std_std",
    "eeg.resting.PowerSpectralDensity_alpha_std_std",
    "eeg.resting.PowerSpectralDensity_alphan_std_std",
    "eeg.resting.PowerSpectralDensity_beta_std_std",
    "eeg.resting.PowerSpectralDensity_betan_std_std",
    "eeg.resting.PowerSpectralDensity_gamma_std_std",
    "eeg.resting.PowerSpectralDensity_gamman_std_std",
    "eeg.resting.PowerSpectralDensity_summary_se_std_std",
    "eeg.resting.PowerSpectralDensitySummary_summary_msf_std_std",
    "eeg.resting.PowerSpectralDensitySummary_summary_sef90_std_std",
    "eeg.resting.PowerSpectralDensitySummary_summary_sef95_std_std",
    "eeg.resting.PermutationEntropy_default_std_std",
    "eeg.resting.SymbolicMutualInformation_weighted_std_std",
    "eeg.resting.KolmogorovComplexity_default_std_std",
]
AGESEX_FEATURES = ["age", "sex"]

DEATH_FEATURES = ["death", "death.cause"]
DIAG_FEATURES = ["doc.enrol", "doc.disch"]
DIAGBIN_FEATURES = ["doc.enrol.bi", "doc.disch.bi"]

TARGETS = [
    # "GOS-E.3",
    # "GOS-E.3.bin",
    # "GOS-E.12",
    "GOS-E.12.bin",
    # "mRS.3.bin",
    "mRS.12.bin",
    # "CPC.3.bin",
    "CPC.12.bin",
]

abcd_map = {"A": 1, "B": 2, "C": 3, "D": 4, "nonABCD": np.nan}
doc_bi_map = {
    "coma/UWS": 0,
    "nonUWS": 1,
}

sex_map = {
    "Male": 0,
    "Female": 1,
}

doc_map = {
    "Coma": 1,
    "UWS": 2,
    "Unsure/UWS": 2,
    "MCS-": 3,
    "MCS+": 4,
    "CS": 5,
    "eMCS": 5,
    "LIS": 6,
    "Conscious": 7,
}

outcome_bin = {"Bad": 0, "Good": 1, "No data": np.nan}

outcome_map = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "No data": np.nan,
}

death_map = {
    "No": 0,
    "Yes": 1,
}

to_map = {
    "sex": sex_map,
    "eeg.abcd": abcd_map,
    "doc.enrol": doc_map,
    "doc.disch": doc_map,
    "doc.enrol.bi": doc_bi_map,
    "doc.disch.bi": doc_bi_map,
    "GOS-E.3": outcome_map,
    "GOS-E.12": outcome_map,
    "GOS-E.3.bin": outcome_bin,
    "GOS-E.12.bin": outcome_bin,
    "mRS.3.bin": outcome_bin,
    "mRS.12.bin": outcome_bin,
    "CPC.3.bin": outcome_bin,
    "CPC.12.bin": outcome_bin,
    "death": death_map,
}


ALL_FEATURES = (
    DIAG_FEATURES +
    FMRI_FEATURES +
    EEG_VISUAL_FEATURES +
    EEG_ABCD_FEATURES +
    EEG_MODEL_FEATURES +
    EEG_STIM_FEATURES +
    EEG_RESTING_FEATURES
)