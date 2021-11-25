from __future__ import annotations

import functools
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from glob import glob
from itertools import chain
from operator import attrgetter
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
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
    @functools.lru_cache(maxsize=1)
    def get_slices(patch_idx):
        return np.asarray([
            np.mgrid[tuple(slice(*map(int, d)) for d in dim)]
            for dim in (torch.cat([patch_idx[None], patch_idx[None]+1]).T.swapaxes(0, 1) * patch_size)
        ])

    def cast_to_lowest_dtype(array: np.ndarray) -> np.ndarray:
        return array.astype(
            bool
            if (array_min := array.min()) == 0 and (array_max := array.max()) == 1
            else np.result_type(*map(np.min_scalar_type, (array_min, array_max)))
        )

    arrays, counts, patch_size = {}, {}, None

    for ret_values in run_eval(model, dataset):
        for (encodings, names, img_idx, patch_idx) in ret_values:

            if patch_size is None:
                patch_size = torch.as_tensor(encodings.shape[1:])

            slices = get_slices(patch_idx)

            u_names, u_idx, u_counts = np.unique(
                names, return_counts=True, return_index=True
            )

            for name, image_index, count in zip(u_names, img_idx[u_idx], u_counts):
                current_count = counts.setdefault(name, np.asarray(dataset._lengths[image_index]))
                current_array = arrays.setdefault(name, torch.empty(
                    size=tuple(patch_size*dataset._sizes[image_index]),
                    dtype=encodings.dtype,
                    device=encodings.device
                ))

                mask = img_idx == image_index
                current_array[slices[mask].swapaxes(0, 1)] = encodings[mask]
                current_count -= count  # persistent because of np.array

                if current_count == 0:
                    counts.pop(name)
                    yield name, cast_to_lowest_dtype(arrays.pop(name).cpu().numpy())


@torch.no_grad()
def run_eval(model, dataset, batch_size=2500):

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=18,
        prefetch_factor=10
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')

    model = model.to(device)
    model.eval()

    def extract_path(path: str) -> str:
        return Path(path).parent.stem + '/' + Path(path).stem

    max_pool = None

    logging.info("Setup complete, starting encoding")

    for imgs, labels, (img_index, patch_index, img_path, label_path) in dataloader:

        imgs, labels = (
            imgs.to(device, non_blocking=True),
            labels.to(device, non_blocking=True)
        )

        with torch.autocast('cuda'):
            encodings, encoding_indices, encoding_loss = tuple(zip(*model.encoder(imgs)))[0]

        if max_pool is None:
            max_pool = partial(F.adaptive_max_pool2d, output_size=encoding_indices.shape[-2:])

        labels_pooled = max_pool(labels.to(torch.half)).to(labels.dtype).squeeze()

        yield (
            (data, list(map(extract_path, paths)), img_index, patch_index)
            for data, paths in (
                (encoding_indices, img_path),
                (labels_pooled, label_path)
            )
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
    del model.decoder  # don't need the decoder

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
        for array_name, array in tqdm.tqdm(
            get_encodings(model, dataset),
            total=len(dataset._lengths) * 2  # FIXME: remove hardcoded 2
        ):
            out_path = ckpt_folder / 'encodings' / (array_name + '.npy')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), array)


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
