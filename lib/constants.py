import numpy as np


_copenhagen_montages = {
    '25': 'standard_1020',
    '19': 'standard_1020'
}

_copenhagen_ch_names = {
    '25': ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7',
           'F8', 'T7', 'T8', 'P7', 'P8', 'T9', 'T10', 'Fz', 'Cz', 'Pz', 'F10',
           'F9', 'P9', 'P10'],
    '19': ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7',
           'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
}


_copenhagen_25_rois = {
    'p3a': np.array([2, 3, 18, 19, 4, 5]),  # Fz + Cz
    'p3b': np.array([19, 4, 5, 20, 6, 7]),  # Cz + Pz
    'mmn': np.array([2, 3, 18, 19, 4, 5]),  # Fz + Cz
    'cnv': np.array([2, 3, 18, 19, 4, 5]),  # Fz + Cz
    'Fz': np.array([2, 3, 18]),
    'Cz': np.array([19, 4, 5]),
    'Pz': np.array([20, 6, 7]),
    'scalp': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       16, 17, 18, 19, 20, 21, 22, 23, 24]),
    'nonscalp': None
}

_copenhagen_19_rois = {
    'p3a': np.array([2, 3, 16, 17, 4, 5]),  # Fz + Cz
    'p3b': np.array([17, 4, 5, 18, 6, 7]),  # Cz + Pz
    'mmn': np.array([2, 3, 16, 17, 4, 5]),  # Fz + Cz
    'cnv': np.array([2, 3, 16, 17, 4, 5]),  # Fz + Cz
    'Fz': np.array([2, 3, 16]),
    'Cz': np.array([17, 4, 5]),
    'Pz': np.array([18, 6, 7]),
    'scalp': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       16, 17, 18]),
    'nonscalp': None
}


_egi256__copenhagen25_map = {
    'E37': 'Fp1',
    'E18': 'Fp2',
    'E36': 'F3',
    'E224': 'F4',
    'E59': 'C3',
    'E183': 'C4',
    'E87': 'P3',
    'E153': 'P4',
    'E116': 'O1',
    'E150': 'O2',
    'E47': 'F7',
    'E2': 'F8',
    'E69': 'T7',
    'E202': 'T8',
    'E96': 'P7',
    'E170': 'P8',
    'E68': 'T9',
    'E210': 'T10',
    'E21': 'Fz',
    'E81': 'Cz',
    'E101': 'Pz',
    'E226': 'F10',
    'E252': 'F9',
    'E106': 'P9',
    'E169': 'P10',
}

_egi256__copenhagen19_map = {
    'E37': 'Fp1',
    'E18': 'Fp2',
    'E36': 'F3',
    'E224': 'F4',
    'E59': 'C3',
    'E183': 'C4',
    'E87': 'P3',
    'E153': 'P4',
    'E116': 'O1',
    'E150': 'O2',
    'E47': 'F7',
    'E2': 'F8',
    'E69': 'T7',
    'E202': 'T8',
    'E96': 'P7',
    'E170': 'P8',
    'E21': 'Fz',
    'E81': 'Cz',
    'E101': 'Pz',
}
