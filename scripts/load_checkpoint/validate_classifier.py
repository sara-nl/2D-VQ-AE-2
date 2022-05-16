import logging
import os
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose, initialize
from hydra.utils import instantiate
from torchmetrics import MetricCollection, Precision, Recall

from validation_nn.model import CNNClassifier


def load_classifier(cfg_path, model_ckpt_path):
    cfg = OmegaConf.load(cfg_path)

    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(model_ckpt_path)['state_dict'])
    model.cuda()

    cfg.datamodule.val_dataloader_conf.batch_size = 1
    val_dataloader = instantiate(cfg.datamodule).val_dataloader()

    metrics = MetricCollection(
        Precision(num_classes=3, ignore_index=0, average='none', mdmc_average='global'),
        Recall(num_classes=3, ignore_index=0, average='none', mdmc_average='global')
    ).cuda()

    metrics_out, batches = [], []

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            for batch, labels in val_dataloader:
                batch, labels = batch.cuda(), labels.cuda()

                out = model(batch)

                # XXX: hack
                out[:, 0][(labels == 0)] = torch.inf

                batches.append((batch.cpu(), labels.cpu(), out.cpu()))
                metrics_out.append(metrics(out, labels))

    precision, recall = (
        torch.stack([metric['Precision'].cpu() for metric in metrics_out]),
        torch.stack([metric['Recall'].cpu() for metric in metrics_out])
    )

    #mean_precision, mean_recall = (
    #    precision.mean(dim=0),
    #    recall.mean(dim=0),
    #)
    breakpoint()
    for i, (batch, labels, out) in enumerate(batches):
        print(f'batch_{i}')

        if i == 5:
            for j in range(batch.max()):
                img = batch[0, 0].clone()
                mask = img == j

                img[~mask] = 0
                img[mask] = 1

                imsave_path = Path(f'imgs/batch_{i}/embedding_{j}.bmp')
                imsave_path.parent.mkdir(parents=True, exist_ok=True)
                plt.imsave(imsave_path, img)
        else:
            continue

            plt.imsave(f'imgs/batch_{i}.png', batch[0,0])
            plt.imsave(f'imgs/labels_{i}.png', labels[0])
            plt.imsave(f'imgs/pred_{i}.png', out[0].argmax(dim=0))



if __name__ == '__main__':
    from utils.conf_helpers import add_resolvers

    add_resolvers()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    os.environ["CAMELYON16_EMBEDDINGS_PATH"] = "/home/robertsc/2D-VQ-AE-2/vq_ae/multirun/2021-11-01/13-43-06/1/encodings.hdf5"

    cfg_path =         "/home/robertsc/2D-VQ-AE-2/validation_nn/multirun/2022-04-20/16-14-51/0/experiment.yml"
    model_checkpoint = "/home/robertsc/2D-VQ-AE-2/validation_nn/multirun/2022-04-20/16-14-51/0/lightning_logs/version_0/checkpoints/epoch=902-step=109263.ckpt"

    load_classifier(cfg_path, model_ckpt_path=model_checkpoint)
