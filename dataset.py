from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset


__all__ = ['ABCDMidDataset', 'ABCDFrameDataset']


CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])


def generate_filename_dataset(root_dir):
    df = pd.DataFrame(None, columns=['subject', 'file_mid'])
    subj_dirs = [p for p in Path(root_dir).glob('sub-*') if p.is_dir()]

    for subj_dir in subj_dirs:
        fns_mid = sorted(subj_dir.glob('ses-*/func/*task-mid*bold.nii*'))

        for fn_mid in fns_mid:
            df = df.append(pd.Series({
                'subject': subj_dir.name.split('-')[1],
                'file_mid': str(fn_mid)
            }), ignore_index=True)

    return df


def generate_subject_dataset(root_dir):
    subj_dirs = [p for p in root_dir.glob('sub-*') if p.is_dir()]
    subjects = [p.name.split('-')[1] for p in subj_dirs]
    return pd.DataFrame({'subject': subjects})


def load_nifti_to_numpy(filename):
    img = nib.load(str(filename))
    return np.array(img.dataobj)


class ABCDMidDataset(Dataset):
    def __init__(self, root_dir, subject_wise=True):
        self.root_dir = root_dir
        self.subject_wise = subject_wise
        if subject_wise:
            self.df = generate_subject_dataset(root_dir)
        else:
            self.df = generate_filename_dataset(root_dir)

    def __len__(self):
        return len(self.df)

    def _getitem_filename(self, idx):
        row = self.df.iloc[idx]

        subject = row['subject']
        file_mid = row['file_mid']

        arr_mid = load_nifti_to_numpy(file_mid)

        return {'subject': subject, 'file_mid': file_mid, 'mid': arr_mid}

    def _getitem_subject(self, idx):
        row = self.df.iloc[idx]
        subject = row['subject']

        subj_dir = Path(self.root_dir) / ('sub-' + subject)
        file_mid = np.random.choice(
            [*subj_dir.glob('ses-*/func/*task-mid*bold.nii*')])

        arr_mid = load_nifti_to_numpy(file_mid)

        return {'subject': subject, 'file_mid': file_mid, 'mid': arr_mid}

    def __getitem__(self, idx):
        if self.subject_wise:
            return self._getitem_subject(idx)
        else:
            return self._getitem_filename(idx)


class ABCDFrameDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = sorted(Path(root_dir).glob('*.npy'))

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        filename = self.list_files[idx]
        arr = np.load(filename)
        arr = np.pad(arr, [[0, 0], [0, 0], [2, 2]], mode='edge').shape
        arr = np.expand_dims(arr, axis=0)
        return arr
