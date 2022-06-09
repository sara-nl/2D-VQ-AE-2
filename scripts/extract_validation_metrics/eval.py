from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torchmetrics.image
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import call, instantiate
from torch import nn


@hydra.main(config_path="./conf", config_name="config")
def main(checkpoint_dirs):

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=checkpoint_dirs.hydra_config_name):
        experiment = compose(config_name='config')  # 'config' is the actual name of the yaml file


    torch.cuda.empty_cache()

    if 'utils' in experiment:
        call(experiment.utils)

    trainer: pl.Trainer = instantiate(experiment.trainer)


    experiment.train_datamodule.val_dataloader_conf.batch_size = 1
    train_datamodule: pl.LightningDataModule = instantiate(experiment.train_datamodule)
    model: pl.LightningModule = instantiate(
        experiment.model,
        metrics=torchmetrics.MetricCollection([
#            torchmetrics.MeanSquaredError(),
#            torchmetrics.PeakSignalNoiseRatio(),
            torchmetrics.StructuralSimilarityIndexMeasure(),
        ])
    )

    for i, path in enumerate((Path(checkpoint_dirs.checkpoint_dir) / 'lightning_logs').iterdir()):
        assert i != 1, f'more than one folder found in the lightning_logs folder, please check {path}'
        ckpt_path = str([*(path / 'checkpoints').iterdir()][-1])

    with torch.no_grad(), torch.cuda.amp.autocast():
        trainer.validate(model=model, ckpt_path=ckpt_path, datamodule=train_datamodule)


    # return trainer.callback_metrics['val_recon_loss'].item()


if __name__ == '__main__':
    from utils.conf_helpers import add_resolvers
    add_resolvers()

    main()
