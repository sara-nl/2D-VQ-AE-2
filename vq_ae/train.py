
import torch
import pytorch_lightning as pl
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

import utils.conf_helpers # import adds parsers to hydra parser


@hydra.main(config_path="../conf", config_name="camelyon16_config")
def main(experiment):
    torch.cuda.empty_cache()
    
    print(f"found config: {OmegaConf.to_yaml(experiment)}")

    # seed everything
    # callbacks

    trainer: pl.Trainer = instantiate(experiment.trainer)
    model: pl.LightningModule = instantiate(experiment.model)
    train_datamodule: pl.LightningDataModule = instantiate(experiment.train_datamodule)

    trainer.fit(model, train_datamodule)

if __name__ == '__main__':
    main()