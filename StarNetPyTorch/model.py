import torch
import torch.nn as nn
import torch.nn.functional as F


class TFBatchNorm2d(nn.Module):
    """BatchNorm compatible with the original TF graph's training=True usage."""

    def __init__(self, channels, eps=1e-5, momentum=0.99):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, x):
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(0, 2, 3), keepdim=True)
        if self.training:
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(
                    mean.reshape(-1), alpha=1.0 - self.momentum
                )
                self.running_var.mul_(self.momentum).add_(
                    var.reshape(-1), alpha=1.0 - self.momentum
                )
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        return (x - mean) * torch.rsqrt(var + self.eps) * weight + bias


class Generator(nn.Module):
    def __init__(self, input_channels=3, batch_norm_eps=1e-5):
        super().__init__()
        filters = [64, 128, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 128, 64]

        self.down_convs = nn.ModuleList()
        in_channels = input_channels
        for out_channels in filters[:8]:
            self.down_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            )
            in_channels = out_channels

        self.down_bns = nn.ModuleList(
            [TFBatchNorm2d(channels, eps=batch_norm_eps) for channels in filters[1:8]]
        )

        self.up_convs = nn.ModuleList()
        up_in_channels = [
            filters[7],
            filters[8] + filters[6],
            filters[9] + filters[5],
            filters[10] + filters[4],
            filters[11] + filters[3],
            filters[12] + filters[2],
            filters[13] + filters[1],
            filters[14] + filters[0],
        ]
        up_out_channels = filters[8:] + [input_channels]
        for in_channels, out_channels in zip(up_in_channels, up_out_channels):
            self.up_convs.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            )

        self.up_bns = nn.ModuleList(
            [TFBatchNorm2d(channels, eps=batch_norm_eps) for channels in filters[8:15]]
        )
        self.apply(_initialize_layer)

    def forward(self, x):
        original = x
        layers = []

        y = self.down_convs[0](x)
        layers.append(y)

        for i in range(1, 8):
            y = F.leaky_relu(layers[-1], negative_slope=0.2)
            y = self.down_convs[i](y)
            y = self.down_bns[i - 1](y)
            layers.append(y)

        y = F.relu(layers[-1])
        y = self.up_convs[0](y)
        y = self.up_bns[0](y)
        layers.append(y)

        for up_index, skip_index in enumerate([6, 5, 4, 3, 2, 1], start=1):
            y = torch.cat([layers[-1], layers[skip_index]], dim=1)
            y = F.relu(y)
            y = self.up_convs[up_index](y)
            y = self.up_bns[up_index](y)
            layers.append(y)

        y = torch.cat([layers[-1], layers[0]], dim=1)
        y = F.relu(y)
        y = self.up_convs[7](y)
        y = F.relu(y)
        return original - y


class Discriminator(nn.Module):
    def __init__(self, input_channels=3, batch_norm_eps=1e-3):
        super().__init__()
        filters = [32, 64, 64, 128, 128, 256, 256, 256, 8]
        strides = [1, 2, 1, 2, 1, 2, 1, 2, 2]

        self.convs = nn.ModuleList()
        in_channels = input_channels
        for index, (out_channels, stride) in enumerate(zip(filters, strides)):
            padding = 1 if stride == 1 else 0
            self.convs.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=padding,
                )
            )
            in_channels = out_channels

        self.batch_norms = nn.ModuleList(
            [TFBatchNorm2d(channels, eps=batch_norm_eps) for channels in filters[1:]]
        )
        self.dense = nn.Conv2d(filters[-1], 1, kernel_size=1)
        self.apply(_initialize_layer)

    def forward(self, x):
        features = []

        y = F.leaky_relu(self.convs[0](x), negative_slope=0.2)
        features.append(y)

        for index in range(1, len(self.convs)):
            y = self.convs[index](features[-1])
            y = self.batch_norms[index - 1](y)
            y = F.leaky_relu(y, negative_slope=0.2)
            features.append(y)

        prediction = torch.sigmoid(self.dense(features[-1]))
        return (*features[:8], prediction)


def _initialize_layer(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def create_generator(
    weights_path=None,
    device=None,
    input_channels=3,
    batch_norm_eps=1e-5,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Generator(
        input_channels=input_channels,
        batch_norm_eps=batch_norm_eps,
    ).to(device)
    if weights_path:
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        net.load_state_dict(state_dict)
    net.eval()
    return net


def create_discriminator(
    weights_path=None,
    device=None,
    input_channels=3,
    batch_norm_eps=1e-3,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Discriminator(
        input_channels=input_channels,
        batch_norm_eps=batch_norm_eps,
    ).to(device)
    if weights_path:
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        net.load_state_dict(state_dict)
    net.eval()
    return net
