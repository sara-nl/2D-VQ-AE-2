from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from datamodules.camelyon_16 import CAMELYON16DataModule, CAMELYON16RandomPatchDataSet

if __name__ == '__main__':
    data_dir = Path('/project/robertsc/examode/CAMELYON16/')

    # Dataset test
    parser = ArgumentParser()
    parser = CAMELYON16RandomPatchDataSet.add_dataset_specific_args(parser)
    config = parser.parse_args(['--data-dir', str(data_dir)])

    dataset = CAMELYON16RandomPatchDataSet(config)

    # check for tissue
    datapoint_index, n_tries = 1, 1
    n_success = sum(
        np.any(np.logical_or(dataset[datapoint_index] > 0, dataset[datapoint_index] < 255))
        for _ in range(n_tries)
    )
    print(f'succes: {n_success}/{n_tries}')


    # Datamodule test
    parser = ArgumentParser()
    parser = CAMELYON16DataModule.add_datamodule_specific_args(parser)
    config = parser.parse_args(['--data-dir', str(data_dir), '--num-workers', 0])

    datamodule = CAMELYON16DataModule(config)
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    train_datapoint = next(iter(train_dataloader))
    val_datapoint = next(iter(val_dataloader))
