
import torch
import pytorch_lightning as pl
import hydra
from hydra.utils import instantiate, call

import utils.conf_helpers # import adds parsers to hydra parser

@hydra.main(config_path="../conf", config_name="camelyon16_config")
def main(experiment):
    torch.cuda.empty_cache()
    
    if 'utils' in experiment:
        call(experiment.utils)

    if 'trial' in experiment:
        experiment.trainer.callbacks.pytorch_lightning_pruning_callback.trial = experiment.trial

    trainer: pl.Trainer = instantiate(experiment.trainer)
    model: pl.LightningModule = instantiate(experiment.model)
    train_datamodule: pl.LightningDataModule = instantiate(experiment.train_datamodule)

    trainer.fit(model, train_datamodule)

    return trainer.callback_metrics['val_recon_loss'].item()

if __name__ == '__main__':
    main()