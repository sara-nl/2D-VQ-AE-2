import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import call, instantiate
from omegaconf import OmegaConf


@hydra.main(config_path="../conf", config_name="validation_nn_camelyon16_embeddings")
def main(experiment):
    torch.cuda.empty_cache()

    OmegaConf.save(experiment, 'experiment.yml')

    if 'utils' in experiment:
        call(experiment.utils)

    if 'trial' in experiment:
        experiment.trainer.callbacks.pytorch_lightning_pruning_callback.trial = experiment.trial

    trainer: pl.Trainer = instantiate(experiment.trainer)
    model: pl.LightningModule = instantiate(experiment.model)
    datamodule: pl.LightningDataModule = instantiate(experiment.datamodule)

    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics['val_loss'].item()


if __name__ == '__main__':
    from utils.conf_helpers import add_resolvers
    add_resolvers()

    main()
