import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import call, instantiate
from omegaconf import OmegaConf


@hydra.main(config_path="../conf", config_name="vq_ae_camelyon16_config")
def main(experiment):
    import oneccl_bindings_for_pytorch

    torch.cuda.empty_cache()

    OmegaConf.save(experiment, 'experiment.yml')

    if 'utils' in experiment:
        call(experiment.utils)

    if 'trial' in experiment:
        experiment.trainer.callbacks.pytorch_lightning_pruning_callback.trial = experiment.trial

    trainer: pl.Trainer = instantiate(experiment.trainer)
    model: pl.LightningModule = instantiate(experiment.model)
    train_datamodule: pl.LightningDataModule = instantiate(experiment.train_datamodule)

    # we pass the dataloaders explicitely so we can use memory_format=torch.channels
    trainer.fit(model, train_datamodule.train_dataloader(), train_datamodule.val_dataloader())

    return trainer.callback_metrics['val_recon_loss'].item()


if __name__ == '__main__':
    from utils.conf_helpers import add_resolvers
    add_resolvers()

    main()
