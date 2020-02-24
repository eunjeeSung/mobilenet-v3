import torch
import torch.nn as nn

from mobilenet_v3 import MobileNetV3, mobilenet_v3


if __name__ == '__main__':
    net = mobilenet_v3()
    #print('mobilenetv3:\n', net)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    input_size=(1, 3, 224, 224)
    # pip install --upgrade git+https://github.com/kuan-wang/pytorch-OpCounter.git
    from thop import profile
    #flops, params = profile(net, input_size=input_size)
    #print(flops)
    #print(params)
    #print('Total params: %.2fM' % (params/1000000.0))
    #print('Total flops: %.2fM' % (flops/1000000.0))
    x = torch.randn(input_size)
    out = net(x)
    print(out)