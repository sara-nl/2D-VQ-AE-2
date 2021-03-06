from collections.abc import Sequence
from typing import Callable, Optional, Any, Tuple, List

import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import MetricCollection

from vq_ae.optim.sam import SAM


class CNNClassifier(pl.LightningModule):  # noqa

    def __init__(
        self,
        optim: Callable[[Sequence[torch.Tensor]], torch.optim.Optimizer],
        loss_f: nn.Module,
        layers: nn.Module,
        lr_scheduler: Optional[Callable[[torch.optim.Optimizer], Any]] = None,  # there doesn't exist a
        train_metrics: Optional[MetricCollection] = None,
        val_metrics: Optional[MetricCollection] = None,
        test_metrics: Optional[MetricCollection] = None,
        **kwargs
    ):
        super().__init__()

        for attr_name, attr in (
            ('optim', optim),
            ('loss_f', loss_f),
            ('layers', layers),
            ('lr_scheduler', lr_scheduler),
            ('train_metrics', train_metrics),
            ('val_metrics', val_metrics),
            ('test_metrics', test_metrics),
            *kwargs.items()
        ):
            setattr(self, attr_name, attr)

            def init(*modules):
                for module in modules:
                    if isinstance(module, nn.Conv2d):
                        nn.init.kaiming_normal_(model.weight)
                        nn.init.zeros_(model.bias)


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

        if self.loss_f.reduction == 'sum':
            log_loss = loss / torch.as_tensor(out.shape).prod()
            self.log(
                f'{mode}_loss_unreduced',
                loss,
                prog_bar=(mode == 'train'),
                sync_dist=val_or_test
            )
        else:
            log_loss = loss

        self.log(
            f'{mode}_loss',
            log_loss,
            prog_bar=(mode == 'train'),
            sync_dist=val_or_test
        )

        if (
            hasattr(self, (metric_name := f'{mode}_metrics'))
            and (metrics := getattr(self, metric_name)) is not None
        ):

            # --------------------------------------
            # XXX: HACK
            if out.shape[1] == 1:
                mask = labels != 0
                labels = labels.clone()
                out, labels = (
                    out[mask.unsqueeze(dim=1)],
                    (labels[mask] - 1).bool()
                )
            else:
                out[:, 0][(labels == 0)] = torch.inf
            # ---------------------------------------

            for name, metric in metrics(out, labels).items():
                if metric.ndim > 0 and len(metric) > 1:
                    for i, value in enumerate(metric):
                        self.log(f'{mode}_{name}_{i}', value, prog_bar=(mode == 'train'))
                else:
                    self.log(f'{mode}_{name}', metric, prog_bar=(mode == 'train'))

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

