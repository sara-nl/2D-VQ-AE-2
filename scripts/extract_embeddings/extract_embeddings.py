from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from glob import glob
from itertools import chain
from operator import attrgetter
from pathlib import Path

import numpy as np
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from vq_ae.model import VQAE


@dataclass
class ExtractEmbeddingsConf(DictConfig):
    run_path: str
    dataset_target_hotswap: str
    force_outputs_or_multirun_as_root: bool


@hydra.main(config_path="./conf", config_name="camelyon16_embeddings")
def main(
    cfg: ExtractEmbeddingsConf,
):
    cfg = instantiate(cfg)

    folder_path = _parse_input_path(cfg.run_path)
    run_folders = find_all_run_folders(folder_path, patterns=('.hydra', 'lightning_logs'))

    for run_folder in run_folders:
        checkpoint_path = list(run_folder.rglob('epoch=*.ckpt'))
        if len(checkpoint_path) == 0:
            logging.info(f"Could not find a model checkpoint in {run_folder}")
            continue

        # TODO: smarter checkpoint finder instead of just taking the last checkpoint
        model = VQAE.load_from_checkpoint(sorted(checkpoint_path)[-1])  # type: ignore

        GlobalHydra.instance().clear()
        with initialize_config_dir(str(run_folder / '.hydra')):
            run_cfg = compose('config')

        dataset_config = run_cfg.train_datamodule.train_dataloader_conf.dataset
        dataset_config._target_ = cfg.dataset_target_hotswap
        dataset = instantiate(dataset_config)

        transforms = instantiate(run_cfg)


def find_all_run_folders(
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


def _parse_input_path(input_path: str) -> Path:
    path = (
        pt
        if (pt := Path(input_path)).is_absolute()
        else '../../../' / pt  # remove time, date, 'outputs' folders
    ).resolve()
    assert path.is_dir(), f'resolved path {path} does not seem to be a dir'

    return path


if __name__ == '__main__':
    from utils.conf_helpers import add_resolvers
    add_resolvers()

    main()
