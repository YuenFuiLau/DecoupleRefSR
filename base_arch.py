import sys
import math

import torch
from torch import nn
from torch.nn import functional as F

from op import fused_leaky_relu, upfirdn2d

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv0 = EqualConv2d(
            in_channel=3,
            out_channel=64,
            kernel_size=1
            )

        self.conv1 = EncoderLayer(
            in_channel=64,
            out_channel=128,
            kernel_size=3,
            downsample=True
        )

        self.conv2 = EncoderLayer(
            in_channel=128,
            out_channel=256,
            kernel_size=3,
            downsample=True
        )

        self.conv3 = EncoderLayer(
            in_channel=256,
            out_channel=512,
            kernel_size=3,
            downsample=True
        )

        self.conv4 = EncoderLayer(
            in_channel=512,
            out_channel=512,
            kernel_size=3,
            downsample=True
        )

    def forward(self, x):

        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)

        return y

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv0 = EqualConv2d(
            in_channel=64,
            out_channel=3,
            kernel_size=1
            )

        self.conv1 = DecoderLayer(
            in_channel=128,
            out_channel=64,
            kernel_size=3,
            upsample=True
        )

        self.conv2 = DecoderLayer(
            in_channel=256,
            out_channel=128,
            kernel_size=3,
            upsample=True
        )

        self.conv3 = DecoderLayer(
            in_channel=512,
            out_channel=256,
            kernel_size=3,
            upsample=True
        )

        self.conv4 = DecoderLayer(
            in_channel=512,
            out_channel=512,
            kernel_size=3,
            upsample=True
        )

    def forward(self, x):

        y = self.conv4(x)
        y = self.conv3(y)
        y = self.conv2(y)
        y = self.conv1(y)
        y = self.conv0(y)
        
        return y

class EncoderLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        super().__init__()

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

            stride = 2
            padding = 0

        else:
            self.blur = None
            stride = 1
            padding = kernel_size // 2

        
        self.conv = EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=padding,
                stride=stride,
                bias=bias and not activate,
            )

        self.activate = nn.LeakyReLU(negative_slope=0.2) if activate else None

    def forward(self, input):
        out = self.blur(input) if self.blur is not None else input
        out = self.conv(out) 
        out = self.activate(out) if self.activate is not None else out
        return out


class DecoderLayer(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,    
    ):
        super().__init__()
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)
            self.conv = EqualTransposeConv2d(
                in_channel, 
                out_channel, 
                kernel_size, 
                stride=2, 
                padding=0, 
                bias=bias and not activate,
            )
        else:
            self.conv = EqualConv2d(
                in_channel, 
                out_channel, 
                kernel_size, 
                stride=1, 
                padding=kernel_size//2,
                bias=bias and not activate,
            )
            self.blur = None

        self.activate = nn.LeakyReLU(negative_slope=0.2) if activate else None

    def forward(self, input):
        out = self.conv(input)
        out = self.blur(out) if self.blur is not None else out   

        out = self.activate(out.contiguous()) if self.activate is not None else out
        
        return out        


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualTransposeConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        weight = self.weight.transpose(0,1)
        out = F.conv_transpose2d(
            input,
            weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )

class ToRGB(nn.Module):
    def __init__(
        self, 
        in_channel, 
        upsample=True, 
        blur_kernel=[1, 3, 3, 1]
        ):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)
        self.conv = EqualConv2d(in_channel, 3, 3, stride=1, padding=1)

    def forward(self, input, skip=None):
        out = self.conv(input)
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )        

class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(nn.LeakyReLU(negative_slope=0.2))

        super().__init__(*layers)


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


if __name__ == "__main__":

    device = "cuda"
    model = Encoder().to(device)
    demodel = Decoder().to(device)
    x = torch.randn(1,3,512,512).to(device)

    feat = model(x)
    img = demodel(feat)
    print(feat.shape)
    print(img.shape)