import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mapping import mapping
import net_pb2 as n
import gzip
import onnx
from binary_reader import BinaryReader

net = n.Net()

f = open("T79", "rb")
weights = gzip.decompress(f.read())
net.ParseFromString(weights)
f.close()


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel, se_channels):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.network = nn.Sequential(
            nn.Linear(channel, se_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(se_channels, 2 * channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.avg_pool(inputs)
        x = x.view(b, c)
        x = self.network(x)
        x = x.view(b, c, 1, 1)
        x = inputs * x
        return x


class ConvBlock(nn.Module):

    def __init__(
            self, in_channels: int, out_channels: int, kernel_size: int, relu: bool = True
    ):
        super().__init__()
        # we only support the kernel sizes of 1 and 3
        assert kernel_size in (1, 3)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=1 if kernel_size == 3 else 0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.beta = nn.Parameter(torch.zeros(out_channels))  # type: ignore
        self.relu = relu

        # initialisation (remove once weight loading is successful)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x += self.beta.view(1, self.bn.num_features, 1, 1).expand_as(x)
        return F.relu(x, inplace=True) if self.relu else x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 3)
        self.conv2 = ConvBlock(out_channels, out_channels, 3, relu=False)
        self.se = SqueezeAndExcitation(out_channels, 32)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = SqueezeAndExcitation(out)
        out += identity
        return F.relu(out, inplace=True)


class Network(nn.Module):
    def __init__(self, in_channels: int, residual_channels: int,
                 residual_layers: int,
                 ):
        super().__init__()
        self.conv_input = ConvBlock(in_channels, residual_channels, 3)
        self.residual_tower = nn.Sequential(
            *[
                ResBlock(residual_channels, residual_channels)
                for _ in range(residual_layers)
            ]
        )
        self.residual_tower = nn.Sequential()
        self.policy_conv = ConvBlock(residual_channels, residual_channels, 1)
        self.policy_conv2 = ConvBlock(residual_channels, 80, 1)
        self.policy_fc = nn.Linear(80, 1858)

        self.value_conv = ConvBlock(residual_channels, 32, 1)
        self.value_conv2 = ConvBlock(32, 128, 1)
        self.value_fc = nn.Linear(128, 3)

    def forward(self, planes):
        # first conv layer
        x = self.conv_input(planes)

        # residual tower
        x = self.residual_tower(x)

        # policy head
        pol = self.policy_conv(x)
        pol = self.policy_conv2(pol)
        pol = torch.flatten(pol, start_dim=1)
        pol1 = np.zeros(1858)
        for i in mapping:
            if i == -1:
                pass
            else:
                pol1[i] = pol[np.where(pol == i)]

        # value head
        val = self.value_conv(x)
        val = self.value_conv2(val)
        val = F.relu(val)
        val = F.softmax(self.value_fc(torch.flatten(val, start_dim=1)), inplace=True)
        return pol1, val

    def step(self, planes):
        pass
        pred_move, pred_val = self(planes)

    def weight_initialisation(self):
        pass


chess = Network(112, 192, 15)
#model = net.onnx_model.model
# onnx.checker.check_model(model)
pls = onnx.load("192x15-2022_0521_0906_54_491.pb")
# print(type(net.weights.input.weights))
# p = UnicodeDammit(net.weights.input.weights.params)
# print(p.original_encoding)
#print(net.weights.input.weights.params)
#input_weights = BinaryReader(net.weights.input.weights.params)
input_layer_bytes = np.array(list(net.weights.input.weights.params), dtype=np.uint8)
data = input_layer_bytes.view(dtype=np.float16)
input_layer_weights = torch.from_numpy(np.reshape(data, (112, 192, 9)))
data_bytes = np.array(list(net.weights.input.weights.params), dtype=np.uint8)
data = data_bytes.view(dtype=np.float16)
# h=net.weights.input.weights.params.decode('big5', errors= 'ignore')
# print(h)