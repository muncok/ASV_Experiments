import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import ResNet 

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        # normalization
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = cos_theta.data.acos()
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta, phi_theta)
        return output # size=(B,Classnum,2)


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
        if "tar" in filename:
            state_dict =  torch.load(filename)['state_dict']
        else:
            state_dict = torch.load(filename)

        self.load_state_dict(state_dict)
        print("model loaded from {}".format(filename))

    def load_extractor(self, filename):
        if "tar" in filename:
            state_dict =  torch.load(filename)['state_dict']
        else:
            state_dict = torch.load(filename)

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

class ResNet34_cr(ResNet):
    """
        remove maxpooling and keep fc
        change first conv's kernel_size 7 --> 3
    """
    def __init__(self, config, inplanes=64, n_labels=10):
        super(ResNet, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(1, inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, inplanes, 3)
        self.layer2 = self._make_layer(BasicBlock, 2*inplanes, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 4*inplanes, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 8*inplanes, 3, stride=2)
        self.fc = nn.Linear(8*inplanes * BasicBlock.expansion, n_labels)

    def load(self, filename):
        self.load_state_dict(torch.load(filename)['state_dict'])
        print("loaded from {}".format(filename))

    def embed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x,2)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.fc(x)

        return x

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

