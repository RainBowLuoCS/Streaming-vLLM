import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3dLinear(nn.Module):
    """Conv3d that uses F.linear when kernel_size == stride (no overlap)."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, bias=True):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.enable_linear = (kernel_size == stride)
        # self.enable_linear = False
        self.input_size = in_channels * math.prod(kernel_size)

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        if self.enable_linear and x.dim() == 5:
            B, C, T, H, W = x.shape
            K1, K2, K3 = self.kernel_size
            x = x.unfold(2, K1, K1).unfold(3, K2, K2).unfold(4, K3, K3)
            x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(-1, self.input_size)
            x = F.linear(x, self.weight.view(self.out_channels, self.input_size), self.bias)
            T2, H2, W2 = T // K1, H // K2, W // K3
            return x.view(B, T2, H2, W2, self.out_channels).permute(0, 4, 1, 2, 3)
        elif self.enable_linear and x.dim() == 2:
            return F.linear(x, self.weight.view(self.out_channels, self.input_size), self.bias)
        return F.conv3d(x, self.weight, self.bias, stride=self.stride)