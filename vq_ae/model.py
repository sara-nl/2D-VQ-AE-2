from typing import Optional, Sequence, Tuple, Union  # Sequence deprecated from python 3.9+

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch import nn, Tensor
from torchvision.utils import make_grid

from utils.conf_helpers import ModuleConf, OptimizerConf
from utils.train_helpers import maybe_repeat_layer
from vq_ae.optim.sam import SAM


class VQAE(pl.LightningModule):  # noqa

    # Optimizer needs runtime self.parameters(), so need to pass conf objects
    def __init__(
        self,
        optim_conf: OptimizerConf,
        loss_f_conf: ModuleConf,
        encoder_conf: ModuleConf,
        decoder_conf: ModuleConf,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # torch.autograd.set_detect_anomaly(True)
        self.optim_conf = optim_conf

        # init rest of the configs
        for attr_name, attr_conf in (
            # optim_conf not present
            ('loss_f', loss_f_conf),
            ('encoder', encoder_conf),
            ('decoder', decoder_conf),
        ):
            setattr(self, attr_name, instantiate(attr_conf))

        for key, value in kwargs.items():
            setattr(self, key, value)

    def forward(self, data: Tensor) -> Tuple[
        Tensor,
        Sequence[Tensor]
    ]:
        encodings, *_, encoding_loss = self.encoder(data)
        out = self.decoder(encodings)

        return out, encoding_loss

    def configure_optimizers(self):
        optim = instantiate(self.optim_conf, params=self.parameters())
        return optim

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode='val')

    def shared_step(self, batch, batch_idx, mode='train'):
        assert mode in ('train', 'val', 'test')
        val_or_test = mode in ('val', 'test')

        out, recon_loss, encoding_loss = (
            self.sam_step_and_update(batch)  # also calls optim.step()
            if isinstance(self.optimizers(), SAM) and not val_or_test
            else self.step(batch)
        )

        if val_or_test and batch_idx == 0:
            n_img = min(5, batch.shape[0])
            self.logger.experiment.add_image(
                tag=f'{mode}_recon_images',
                img_tensor=make_grid(
                    torch.cat((batch[:n_img], out[:n_img]), dim=0),
                    nrow=n_img,
                    normalize=True
                ),
                global_step=self.global_step,
            )

        self.log(
            f'{mode}_recon_loss',
            recon_loss,
            prog_bar=(mode == 'train'),
            sync_dist=val_or_test
        )
        for i, l in enumerate(encoding_loss):
            self.log(f'{mode}_encoding_loss_{i}', l, sync_dist=val_or_test)

        return recon_loss + sum(encoding_loss)

    def sam_step_and_update(self, batch: Tensor) -> Tuple[
        Tensor,  # output image
        Tensor,  # recon loss
        Sequence[Tensor]  # encoding loss
    ]:
        # TODO: find a way to move this to SAM
        assert self.automatic_optimization is False
        optimizer = self.optimizers()
        assert isinstance(optimizer, SAM)

        def sam_step():  # not a closure because we want the output for logging
            out, recon_loss, encoding_loss = self.step(batch)
            self.manual_backward(recon_loss + sum(encoding_loss))
            return out, recon_loss, encoding_loss

        return optimizer.lightning_step(model=self, forward=sam_step)

    def step(self, batch: Tensor) -> Tuple[
        Tensor,  # output image
        Tensor,  # recon loss
        Sequence[Tensor]  # encoding loss
    ]:
        out, encoding_loss = self.forward(batch)
        recon_loss = self.loss_f(input=out, target=batch)

        return out, recon_loss, encoding_loss


class Encoder(nn.Module):
    def __init__(
        self,
        stem_conf: ModuleConf,
        down_block_conf: Union[ModuleConf, Sequence[ModuleConf]],
        n_pre_enc_layers: Union[int, Sequence[int]],
        vq_conf: Sequence[ModuleConf],
        conv_block_conf: ModuleConf,
        shortcut_block_conf: Optional[Union[ModuleConf, Sequence[ModuleConf]]],
    ):
        super().__init__()

        self.in_stem = instantiate(stem_conf)
        vq_layers = instantiate(vq_conf)

        n_pre_enc_layers, down_block_conf, shortcut_block_conf = (
            maybe_repeat_layer(n_pre_enc_layers, len(vq_layers)),
            maybe_repeat_layer(down_block_conf, len(vq_layers)),
            maybe_repeat_layer(shortcut_block_conf, len(vq_layers) - 1)
        )

        pre_enc_conf = [
            [{**conv_block_conf, **{'mode': 'same'}}] * n_pre_enc
            for n_pre_enc in n_pre_enc_layers
        ]

        down_layers, pre_enc_layers, shortcut_layers = ([] for _ in range(3))

        current_in = self.in_stem.out_channels
        for down_layer, pre_enc_layer, shortcut_layer in zip(
            down_block_conf,
            pre_enc_conf,
            (None, *shortcut_block_conf)  # prepend None to fix shortcut channel size
        ):

            down_block = instantiate(down_layer, in_channels=current_in)
            down_layers.append(down_block)
            current_in = down_block.out_channels

            shortcut_layers.append(
                instantiate(shortcut_layer, in_channels=current_in)
                if shortcut_layer is not None else None
            )

            pre_enc_layers.append(nn.Sequential(*(
                instantiate(layer, in_channels=current_in, out_channels=current_in)
                for layer in pre_enc_layer
            )))

        # delete prepended None & append a None so we can easily zip in forward
        del shortcut_layers[0]
        shortcut_layers.append(None)

        self.down_layers = nn.ModuleList(down_layers)
        self.pre_enc_layers, self.shortcut_layers = (
            nn.ModuleList(reversed(layer))
            for layer in (pre_enc_layers, shortcut_layers)
        )
        self.vq_layers = nn.ModuleList(reversed(vq_layers))

    def forward(self, x: Tensor) -> Tuple[  # completely dependent on VQ-layer output
        Sequence[Tensor],  # encodings
        Sequence[Tensor],  # encoding indices
        Sequence[Tensor],  # encoding loss
    ]:

        # Warning: outputs are in order of low-res to high-res!
        # (because of performance reasons)

        down: Tensor = self.in_stem(x)
        downsampled = reversed([(down := down_layer(down)) for down_layer in self.down_layers])

        # tuple to have nice output type
        # zip(*) to get (enc, *_, loss) each in their own sequence
        out = tuple(zip(*(
            # since the last element of self.shortcut_layers is always None,
            # first iteration is always skipped,
            # and aux is always defined in the local scope of the next iteration.
            # walrus operator doesn't support tuple unpacking, so need to do aux[0]
            (aux := enc(pre_enc(down + (shortcut(aux[0]) if shortcut is not None else 0))))  # noqa
            for down, pre_enc, enc, shortcut in zip(
                downsampled,
                self.pre_enc_layers,
                self.vq_layers,
                self.shortcut_layers,
            )
        )))

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        n_enc_layers: int,  # TODO: find a way to remove this param
        stem_conf: ModuleConf,
        up_block_conf: Union[ModuleConf, Sequence[ModuleConf]],
        n_post_enc_layers: Union[int, Sequence[int]],
        conv_block_conf: ModuleConf,
        shortcut_block_conf: Optional[Union[ModuleConf, Sequence[ModuleConf]]]
    ):
        super().__init__()

        self.out_stem = instantiate(stem_conf)

        n_post_enc_layers, up_block_conf, shortcut_block_conf = (
            maybe_repeat_layer(n_post_enc_layers, n_enc_layers),
            maybe_repeat_layer(up_block_conf, n_enc_layers),
            maybe_repeat_layer(shortcut_block_conf, n_enc_layers - 1)
        )

        post_enc_conf = [[{**conv_block_conf, **{'mode': 'same'}}] * n_post_enc for n_post_enc in
                         n_post_enc_layers]

        up_layers, post_enc_layers, shortcut_layers = ([] for _ in range(3))

        current_out = self.out_stem.in_channels
        for up_layer, post_enc_layer, shortcut_layer in zip(
            up_block_conf,
            post_enc_conf,
            (None, *shortcut_block_conf)  # prepend None to fix shortcut channel size
        ):
            up_block = instantiate(up_layer, out_channels=current_out)
            up_layers.append(up_block)
            current_out = up_block.in_channels

            shortcut_layers.append(
                instantiate(shortcut_layer, out_channels=current_out)
                if shortcut_layer is not None else None
            )

            post_enc_layers.append(nn.Sequential(*(
                instantiate(layer, in_channels=current_out, out_channels=current_out)
                for layer in post_enc_layer
            )))

        # delete prepended None & append a None so we can easily zip in forward
        del shortcut_layers[0]
        shortcut_layers.append(None)

        self.up_layers, self.post_enc_layers, self.shortcut_layers = (
            nn.ModuleList(reversed(layer))
            for layer in (up_layers, post_enc_layers, shortcut_layers)
        )

    def forward(self, x: Sequence[Tensor]) -> Tensor:
        # x should be from low-res to high-res

        prev_up = 0
        for enc, shortcut, post_enc, up in zip(
            x,
            self.shortcut_layers,
            self.post_enc_layers,
            self.up_layers
        ):
            prev_up = up(
                prev_up + post_enc(
                    (0 if shortcut is None else shortcut(aux))  # noqa
                    + (aux := enc)  # define aux after shortcut
                )
            )

        return self.out_stem(prev_up)
