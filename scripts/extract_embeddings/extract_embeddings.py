from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from glob import glob
from itertools import chain, starmap
from operator import attrgetter
from pathlib import Path


import numpy as np
import hydra
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vq_ae.model import VQAE


@dataclass
class ExtractEmbeddingsConf(DictConfig):
    run_path: str
    dataset_target_hotswap: str
    force_outputs_or_multirun_as_root: bool


class CheckpointNotFoundError(ValueError):
    ...


class TooManyCheckpointsError(ValueError):
    ...


def get_encodings(
    model: nn.Module,
    dataset: Dataset
):
    def setdefault_(array_dict: dict, array_count: dict, names, img_idx):
        u_info = np.unique(names, return_index=True, return_counts=True)
        for unique_name, unique_index, unique_count in zip(*u_info):
            if unique_name not in array_dict:
                img_index = img_idx[unique_index]
                array_dict[unique_name] = np.empty(
                    patch_size * (dataset._sizes[img_index]),
                    dtype=encodings.dtype
                )
                array_count[unique_name] = dataset._lengths[img_index]
        return u_info

    def pop_array_if_done_(array_dict, array_count, unique_names, unique_counts):
        return_dict = {}
        for u_name, count in zip(unique_names, unique_counts):
            array_count[u_name] -= count
            if array_count[u_name] == 0:
                array_count.pop(u_name)
                return_dict[u_name] = array_dict.pop(u_name)
            elif array_count[u_name] < 0:
                raise RuntimeError("steps left is less than 0, this shouldn't happen")
        return return_dict

    arrays, counts, patch_size = {}, {}, None
    for encodings, idx, names in run_eval(model, dataset):
        img_idx, patch_idx = idx[0], idx[1:].T
        if patch_size is None:
            patch_size = np.asarray(encodings.shape[1:])

        u_names, _, u_counts = setdefault_(arrays, counts, names, img_idx)

        slices = zip(*(
            tuple(np.s_[start:stop] for start, stop in dim)
            for dim in (np.asarray((patch_idx, patch_idx+1)).T * patch_size)
        ))

        for name, slice_idx, value in zip(names, slices, encodings):
            arrays[name][slice_idx] = value

        yield from pop_array_if_done_(arrays, counts, u_names, u_counts).items()


@torch.no_grad()
@torch.autocast('cuda')
def run_eval(model, dataset, batch_size=75):
    device = torch.device('cuda')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=6
    )

    model = model.to(device)

    def extract_path(path: str) -> str:
        return Path(path).parent.stem + '/' + Path(path).stem

    max_pool = None
    for imgs, labels, (img_index, patch_index, img_path, label_path) in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        encodings, encoding_indices, encoding_loss = tuple(zip(*model.encoder(imgs)))[0]

        if max_pool is None:
            max_pool = partial(F.adaptive_max_pool2d, output_size=encoding_indices.shape[-2:])

        labels_pooled = max_pool(labels.to(torch.float)).type(labels.type())

        yield (
            torch.concat([encoding_indices, labels_pooled.squeeze()]).cpu().numpy(),
            torch.concat([img_index[:, None], patch_index], dim=1).repeat(2, 1).T.cpu().numpy(),
            np.asarray(list(map(extract_path, chain(img_path, label_path))))
        )


@hydra.main(config_path="./conf", config_name="camelyon16_embeddings")
def main(
    cfg: ExtractEmbeddingsConf,
):
    cfg = instantiate(cfg)

    run_path = _parse_input_path(cfg.run_path)
    ckpt_folder = find_ckpt_folder(run_path, pattern=('.hydra', 'lightning_logs'))

    checkpoint_path = list(ckpt_folder.rglob('epoch=*.ckpt'))

    if len(checkpoint_path) == 0:
        raise CheckpointNotFoundError(f"Could not find a model checkpoint in {ckpt_folder}")

    # TODO: smarter checkpoint finder instead of just taking the last checkpoint
    model = VQAE.load_from_checkpoint(sorted(checkpoint_path)[-1])  # type: ignore

    GlobalHydra.instance().clear()
    with initialize_config_dir(str(ckpt_folder / '.hydra')):
        run_cfg = compose('config')

    dataset_config: DictConfig = run_cfg.train_datamodule.train_dataloader_conf.dataset
    dataset_config._target_ = cfg.dataset_target_hotswap

    # XXX: Hotfix
    if (
        'transforms' in dataset_config
        and 'compose' in dataset_config.transforms
    ):
        # Jesus christ OmegaConf is so dumb, just let me set a new value
        dataset_config.transforms.compose.transforms.to_tensor_v2 = {
            **dataset_config.transforms.compose.transforms.to_tensor_v2,
            **{'transpose_mask': True}
        }

    for train_stage in ('train', 'validation', 'test'):  # TODO: replace with Enum
        dataset = instantiate({**dataset_config, **{'train': train_stage}})
        for array_name, array in get_encodings(model, dataset):
            out_path = ckpt_folder / 'encodings' / (array_name + '.npy')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), array)
            logging.info(f"Saved array to {out_path}")


def find_ckpt_folder(path: Path, pattern: Iterable[str]):
    ckpt_folders = find_all_ckpt_folders(path, pattern)

    if len(ckpt_folders) > 1:
        raise TooManyCheckpointsError(f'More than one checkpoint found from root dir {path}')

    return ckpt_folders[0]


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
