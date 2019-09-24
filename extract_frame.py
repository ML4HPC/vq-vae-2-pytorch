import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dataset import ABCDMidDataset


def extract_frames(in_path, out_path):
    in_path = Path(in_path)
    out_path = Path(out_path)

    dataset = ABCDMidDataset(in_path)

    if not out_path.exists():
        out_path.mkdir()

    for row in tqdm(dataset, desc='Image files'):
        arr = np.moveaxis(row['mid'], -1, 0)
        for i in tqdm(range(len(arr)), desc='Frames'):
            np.save(out_path / Path(row['file_mid']).name.replace(
                '_bold.nii', f'_frame-{i:03d}.npy'), arr[i])
        del row, arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str)
    parser.add_argument('out_path',  type=str)

    args = parser.parse_args()

    extract_frames(args.in_path, args.out_path)
