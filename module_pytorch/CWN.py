"""

Centered weight normalization in accelerating training of deep neural networks

ICCV 2017

Authors: Lei Huang
"""
import torch.nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from typing import List
from torch.autograd.function import once_differentiable

__all__ = ['CWN_Conv2d']

#  norm funcitons--------------------------------


class CWNorm(torch.nn.Module):
    def forward(self, weight):
        weight_ = weight.view(weight.size(0), -1)
        weight_mean = weight_.mean(dim=1, keepdim=True)
        weight_ = weight_ - weight_mean
        norm = weight_.norm(dim=1, keepdim=True) + 1e-5
        weight_CWN = weight_ / norm
        return weight_CWN.view(weight.size())

class CWN_Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 NScale=1.414, adjustScale=False, *args, **kwargs):
        super(CWN_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        print('CWN:---NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = CWNorm()
        self.scale_ = torch.ones(out_channels, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.conv2d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

if __name__ == '__main__':
    cwn_ = CWNorm()
    print(cwn_)
    w_ = torch.randn(4, 4, 3, 3)
    w_.requires_grad_()
    y_ = cwn_(w_)
    z_ = y_.view(w_.size(0), -1)
    print(z_.norm(dim=1))

    y_.sum().backward()
    print('w grad', w_.grad.size())

