import logging
from collections.abc import Iterable, Sequence
from glob import glob
from itertools import chain, zip_longest
from operator import attrgetter
from pathlib import Path

import h5py
import hydra
import numpy as np


@hydra.main(config_path="./conf", config_name="convert_camelyon16_embeddings")
def main(
    cfg,
):
    """
    The main reason for this script is to convert the (somewhat) numerous .npy files
    to one 'monolithic' .hdf5 file, which is more easy on the snellius fileserver
    """
    in_path = Path(cfg.run_path).resolve()
    assert in_path.is_dir(), f'{in_path} does not seem to be a valid dir'
    npy_paths = find_all_ckpt_folders(in_path, patterns=['*.npy'])

    common_root, tails = find_common_root(npy_paths)

    with h5py.File(str(common_root) + '.hdf5', 'w') as f:
        for npy_path, group_name in zip(npy_paths, tails):
            subgroup = f.create_group(name=group_name)
            for npy_array in npy_path.iterdir():
                subgroup.create_dataset(npy_array.stem, data=np.load(str(npy_array)))
                logging.info(f"Added to group {subgroup} : array {npy_array.stem}")


def find_common_root(paths: Sequence[Path]) -> tuple[Path, list[str]]:
    common_root = Path()
    parts_iterator = zip_longest(*map(lambda p: p.parts, paths))
    for parts in parts_iterator:
        if len(set(parts)) != 1:
            tails = list(parts)
            break
        common_root /= parts[0]
    else:
        return common_root, ['' for _ in paths]

    tails = list(map(
        lambda x: '/'.join(filter(None, x)),
        zip(parts, *(part for part in parts_iterator))
    ))

    return common_root, tails


def find_all_ckpt_folders(
    folder: Path,
    patterns: Iterable[str]
) -> list[Path]:

    all_valid_folders = list(set(map(attrgetter('parent'), chain.from_iterable(
        map(Path, glob(str(folder / '**' / pattern), recursive=True))
        for pattern in patterns
        # Need to switch to glob.glob for matching folder patterns, because path.rglob
        # apparently doesn't follow symlinks, see https://bugs.python.org/issue33428,
        # what I actually want to use is:
        # folder.rglob(pattern) for pattern in patterns
    ))))

    assert len(all_valid_folders) > 0, (
        f"No valid checkpoint folders were found in path {folder}, "
        f"Check that all folders contain all elements from {patterns}"
    )

    return all_valid_folders


if __name__ == '__main__':
    from utils.conf_helpers import add_resolvers
    add_resolvers()

    main()
