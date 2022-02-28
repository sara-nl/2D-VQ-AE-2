from torch import nn


class SequentialFromKwargs(nn.Sequential):
    def __init__(self, **kwargs):
        super().__init__(*kwargs.values())  # sequential accepts a dict as input


class FlattenAfterEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        def permute(inp):
            b, c, *dim, vec = range(inp.ndim)
            return inp.permute(b, c, vec, *dim)

        def flatten(inp):
            b, c, vec, *dim = inp.shape
            return inp.reshape(b, c*vec, *dim)

        return flatten(permute(batch))

