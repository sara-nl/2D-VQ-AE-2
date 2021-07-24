
import torch
import pytorch_lightning as pl
import hydra
from hydra.utils import instantiate  # for creating objects
from hydra.utils import call  # for calling functions

from conf.experiment.experiment import Experiment

@hydra.main(config_path="../conf", config_name="train_vqae")
def main(experiment: Experiment):
    torch.cuda.empty_cache()

    print(experiment)
    # seed everything
    # callbacks

    trainer: pl.Trainer = instantiate(experiment.trainer)
    model: pl.LightningModule = instantiate(experiment.model)
    train_datamodule: pl.LightningDataModule = instantiate(experiment.train_datamodule)

    trainer.fit(model, train_datamodule)

if __name__ == '__main__':
    main()