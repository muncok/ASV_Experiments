import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

from model import AngleLinear

class ResNet34(nn.Module):
    def __init__(self, config, inplanes=16, n_labels=1000):
        super().__init__()
        layers = [3,4,6,3]
        self.inplanes = inplanes
        self.extractor = nn.Sequential(
            nn.Conv2d(1, inplanes, kernel_size=3, stride=2, padding=3,
                                   bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            self._make_layer(BasicBlock, inplanes, layers[0]),
            self._make_layer(BasicBlock, 2*inplanes, layers[1], stride=2),
            self._make_layer(BasicBlock, 4*inplanes, layers[2], stride=2),
            self._make_layer(BasicBlock, 8*inplanes, layers[3], stride=2)
        )

        loss_type = config["loss"]
        if loss_type == "angular":
            self.classifier = AngleLinear(8*inplanes, n_labels)
        elif loss_type == "softmax":
            self.classifier = nn.Linear(8*inplanes, n_labels)
        else:
            print("not implemented loss")
            raise NotImplementedError
        self.embed_dim = 8*inplanes

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            # classifier does not contain Conv2d or BN2d
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def save(self, filename):
        torch.save(self.state_dict(), filename)
        print("model saved to {}".format(filename))

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
        print("model loaded from {}".format(filename))

    def load_extractor(self, filename):
        state_dict =  torch.load(filename)
        extractor_state_dict = {k:v for k,v in state_dict.items() if
        'extractor' in k}
        classifier_state_dict = {k:v for k,v in self.state_dict().items() if
        'extractor' not in k}
        new_state_dict = {**extractor_state_dict, **classifier_state_dict}
        self.load_state_dict(new_state_dict)
        assert(len(extractor_state_dict) > 0)
        print("extractor loaded from {}".format(filename))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def embed(self, x):
        x = self.extractor(x)
        x = F.avg_pool2d(x,x.shape[-2:])
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.classifier(x)

        return x

class ResNet34_v1(ResNet34):
    """
        additional fc layer before output layer
    """
    def __init__(self, config, inplanes=16, n_labels=1000):
        super().__init__(config, inplanes, n_labels)

        extractor_output_dim = 8*inplanes
        classifier = [nn.Linear(extractor_output_dim,
            extractor_output_dim),
            nn.ReLU(inplace=True)]

        loss_type = config["loss"]
        if loss_type == "angular":
            classifier.append(AngleLinear(extractor_output_dim, n_labels))
        elif loss_type == "softmax":
            classifier.append(nn.Linear(extractor_output_dim, n_labels))
        else:
            print("not implemented loss")
            raise NotImplementedError

        self.classifier = nn.Sequential(*classifier)

class ScaleResNet34(ResNet34):
    def __init__(self, config, inplanes, n_labels=1000, alpha=12):
        super().__init__(config, inplanes, n_labels)
        self.alpha = alpha

    def embed(self, x):
        self.extractor(x)
        x = F.normalize(x)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.alpha * x
        x = self.classifier(x)

        return x

