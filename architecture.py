import torch
import torch.nn as nn


def conv2d(inp, oup, size, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, size, stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        HSwish(inplace=True)
    )

def conv2d_nbn(inp, oup, size, hswish=False):
    if hswish:
        return nn.Sequential(
            nn.Conv2d(inp, oup, size, stride=1, padding=0, bias=False),
            HSwish(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, size, stride=1, padding=0, bias=False)
        )


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        relu6 = nn.ReLU6(inplace=True)
        return x * relu6(x+3) / 6.


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        relu6 = nn.ReLU6(inplace=True)
        return relu6(x+3) / 6.


class SE(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SE, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            HSigmoid(inplace=True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x 
        

class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, size, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()

        assert stride in [1, 2]
        assert size in [3, 5]
        padding = (size-1) // 2
        self.use_res_connect = (stride == 1) and (inp == oup)

        self.conv = nn.Sequential(
            # expansion
            nn.Conv2d(inp, exp, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(exp),
            self.nonlinearity(nl, inplace=True),

            # depthwise convolution
            nn.Conv2d(exp, exp, size, stride, padding, groups=exp, bias=False),
            nn.BatchNorm2d(exp),
            SE(exp),
            self.nonlinearity(nl, inplace=True),

            # projection
            nn.Conv2d(exp, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def nonlinearity(self, nl, inplace=True):
        if nl == 'RE':
            return nn.ReLU(inplace=inplace)
        elif nl == 'HS':
            return HSwish(inplace=inplace)
        else:
            raise Exception('Not Implemented!')

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)