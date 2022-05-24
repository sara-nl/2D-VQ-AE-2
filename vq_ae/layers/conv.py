from torch import nn


class ResizeConv2D(nn.Conv2d):
    def __init__(self, *conv_args, **conv_kwargs):
        super().__init__(*conv_args, **conv_kwargs)
        # important to have align_corners=False
        self.upsample = nn.Upsample(mode='bicubic', scale_factor=2, align_corners=False)

    def forward(self, data):
        return super().forward(self.upsample(data))
