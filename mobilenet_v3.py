import torch
import torch.nn as nn

from architecture import MobileBottleneck, conv2d, conv2d_nbn


class MobileNetV3(nn.Module):
    def __init__(self, dropout=0.8, n_class=1000):
        super(MobileNetV3, self).__init__()
        self.input_channel = 16
        self.last_channel = 1280
        # assert input_size % 32 == 0
        # last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        
        self.mobile_setting = [
            # inp, k, exp, out, se, nl,  s
            [16, 3, 16, 16,  True, 'RE', 2],
            [16, 3, 72, 24, False, 'RE', 2],
            [24, 3, 88, 24, False, 'RE', 1],
            [24, 5, 96, 40,  True, 'HS', 2],
            [40, 5, 240, 40, True, 'HS', 1],
            [40, 5, 240, 40, True, 'HS', 1],
            [40, 5, 120, 48, True, 'HS', 1],
            [48, 5, 144, 48, True, 'HS', 1],
            [48, 5, 288, 96, True, 'HS', 2],
            [96, 5, 576, 96, True, 'HS', 1],
            [96, 5, 576, 96, True, 'HS', 1]
        ]

        self.features = []
        self.classifier = []

        self._build_first_stage()
        self._build_bottlenecks()
        self._build_last_stage()
        
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, n_class)
        )
        
        self._initialize_weights()

    def _build_first_stage(self):
        self.features = [conv2d(3, self.input_channel, 3, 2)]

    def _build_bottlenecks(self):      
        for inp, k, exp, out, se, nl,  s in self.mobile_setting:
            self.features.append(MobileBottleneck(inp, out, k, s, exp, se, nl))

    def _build_last_stage(self):
        self.features.append(conv2d(96, 576, 1, 1))
        self.features.append(nn.AdaptiveAvgPool2d(1))
        self.features.append(conv2d_nbn(576, 1024, 1, hswish=True))
        self.features.append(conv2d_nbn(1024, 1280, 1, hswish=False))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)        

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2) # 왜?
        x = self.classifier(x)
        return x


def mobilenet_v3(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        state_dict = torch.load('mobilenetv3_small_67.4.pth.tar', \
                                map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    return model