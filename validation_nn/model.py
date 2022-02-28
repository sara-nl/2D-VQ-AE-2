from collections.abc import Sequence
from typing import Callable, Optional, Any, Tuple

import torch
from torch import nn
import pytorch_lightning as pl

from vq_ae.optim.sam import SAM

class CNNClassifier(pl.LightningModule):  # noqa

    def __init__(
        self,
        optim: Callable[[Sequence[torch.Tensor]], torch.optim.Optimizer],
        loss_f: nn.Module,
        layers: nn.Module,
        lr_scheduler: Optional[Callable[[torch.optim.Optimizer], Any]] = None,  # there doesn't exist a
        **kwargs
    ):
        super().__init__()

        for attr_name, attr in (
            ('optim', optim),
            ('loss_f', loss_f),
            ('layers', layers),
            ('lr_scheduler', lr_scheduler),
            *kwargs.items()
        ):
            setattr(self, attr_name, attr)

    def configure_optimizers(self):
        return (
            [(initiated_optim := self.optim(self.parameters()))],
            [self.lr_scheduler(initiated_optim)] if self.lr_scheduler is not None else []
        )

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode='val')

    def shared_step(self, batch, batch_idx, mode='train'):
        assert mode in ('train', 'val', 'test')
        val_or_test = mode in ('val', 'test')

        inputs, labels = batch

        out, loss = (
            self.sam_step_and_update(inputs, labels)  # also calls optim.step()
            if isinstance(self.optimizers(), SAM) and not val_or_test
            else self.step(inputs, labels)
        )

        self.log(
            f'{mode}_loss',
            loss,
            prog_bar=(mode == 'train'),
            sync_dist=val_or_test
        )

        if hasattr(self, 'metrics'):
            for name, metric in self.metrics(out, labels).items():
                self.log(f'{mode}_{name}', metric)

        return loss

    def sam_step_and_update(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[
        torch.Tensor,  # predictions
        torch.Tensor,  # loss
    ]:
        # TODO: find a way to move this to SAM
        assert self.automatic_optimization is False
        optimizer = self.optimizers()
        assert isinstance(optimizer, SAM)

        def sam_step():  # not a closure because we want the output for logging
            out, loss = self.step(inputs, labels)
            self.manual_backward(loss)
            return out, loss

        return optimizer.lightning_step(model=self, forward=sam_step)

    def step(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[
        torch.Tensor,  # predictions
        torch.Tensor,  # loss
    ]:

        out = self.forward(inputs)
        classification_loss = self.loss_f(input=out, target=labels)

        return out, classification_loss

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.layers(data)
