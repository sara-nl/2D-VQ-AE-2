import torch
import torch.nn.functional as F
from torch import nn


class EMAVectorQuantizer(nn.Module):
    # Code originally adapted from:
    # https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=fknqLRCvdJ4I
    #
    # Adapted version copied from
    # https://github.com/sara-nl/3D-VQ-VAE-2

    """
    EMA-updated Vector Quantizer
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        decay: float,
        laplace_alpha: float
    ):
        super().__init__()

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embed", embed)  # e_i
        self.register_buffer("embed_avg", embed.clone())  # m_i
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))  # N_i

        # Needs to be a buffer, otherwise doesn't get added to state dict
        self.register_buffer("first_pass", torch.as_tensor(1))

        self.commitment_cost = commitment_cost

        self.decay = decay
        self.laplace_alpha = laplace_alpha

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

    def embed_code(self, embed_idx):
        return F.embedding(embed_idx, self.embed)

    def _update_ema(self, flat_input, encoding_indices):
        # buffer updates need to be in-place because of distributed
        encodings_one_hot = F.one_hot(
            encoding_indices, num_classes=self.num_embeddings
        ).type_as(flat_input)

        new_cluster_size = encodings_one_hot.sum(dim=0)
        dw = encodings_one_hot.T @ flat_input

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(new_cluster_size)
            torch.distributed.all_reduce(dw)

        self.cluster_size.data.mul_(self.decay).add_(
            new_cluster_size, alpha=(1 - self.decay)
        )

        self.embed_avg.data.mul_(self.decay).add_(dw, alpha=(1 - self.decay))

        # Laplacian smoothing
        n = self.cluster_size.sum()
        cluster_size = n * (  # times n because we don't want probabilities but counts
            (self.cluster_size + self.laplace_alpha)
            / (n + self.num_embeddings * self.laplace_alpha)
        )

        embed_normalized = self.embed_avg / cluster_size.unsqueeze(dim=-1)
        self.embed.data.copy_(embed_normalized)

    def _init_ema(self, flat_input):
        mean = flat_input.mean(dim=0)
        std = flat_input.std(dim=0)
        cluster_size = flat_input.size(dim=0)

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(mean)
            torch.distributed.all_reduce(std)
            mean /= torch.distributed.get_world_size()
            std /= torch.distributed.get_world_size()

            cluster_size *= torch.distributed.get_world_size()

        self.embed.mul_(std)
        self.embed.add_(mean)
        self.embed_avg.copy_(self.embed)

        self.cluster_size.data.add_(cluster_size / self.num_embeddings)
        self.first_pass.mul_(0)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, inputs):
        ndim = inputs.dim()
        assert ndim >= 3

        if inputs.shape[1] != self.embedding_dim:
            raise NotImplementedError(
                'VQ dim != channel dim not supported;'
                f' found channel dim of {inputs.shape[1]}, expected {self.embedding_dim}'
            )

        inputs = inputs.float()

        with torch.no_grad():
            channel_last = inputs.permute(
                0,
                *range(2, ndim),
                1
            )  # XXX: might not actually be necessary
            input_shape = channel_last.shape

            flat_input = channel_last.reshape(-1, self.embedding_dim)

            if self.training and self.first_pass:
                self._init_ema(flat_input)

            # although faster, mm is too inaccurate:
            # https://github.com/pytorch/pytorch/issues/42479
            encoding_indices = torch.argmin(
                torch.cdist(
                    flat_input,
                    self.embed,
                    ndim,
                    compute_mode='donot_use_mm_for_euclid_dist'
                )
                , dim=1
            )
            quantized = self.embed_code(encoding_indices).reshape(input_shape)

            if self.training:
                self._update_ema(flat_input, encoding_indices)

            # avg_probs = torch.mean(encodings_one_hot, dim=0)
            # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

            # Cast everything back to the same order and dimensions of the input
            quantized = quantized.permute(0, -1, *range(1, ndim - 1))
            encoding_indices = encoding_indices.reshape(input_shape[:-1])

        # Don't need to detach quantized; doesn't require grad
        e_latent_loss = F.mse_loss(quantized, inputs)
        loss = self.commitment_cost * e_latent_loss

        # Trick to have identity backprop grads
        quantized = inputs + (quantized - inputs).detach()

        # don't change this order without checking everything
        return (
            quantized,
            # perplexity,
            encoding_indices,
            loss,
        )
