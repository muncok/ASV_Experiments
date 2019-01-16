# This codes based on  https://github.com/SiddGururani/Pytorch-TDNN
import torch
import torch.nn as nn
import math

# coding=utf-8
# Copyright 2018 jose.fonollosa@upc.edu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class st_pool_layer(nn.Module):
    def __init__(self):
        super(st_pool_layer, self).__init__()

    def forward(self, x):
        mean = x.mean(2)
        std = x.std(2)
        stat = torch.cat([mean, std], -1)

        return stat

class gTDNN(nn.Module):
    def __init__(self, config, n_labels=31):
        super(gTDNN, self).__init__()
        inDim = config['input_dim']
        self.tdnn = nn.Sequential(
            nn.Conv1d(inDim, 450, stride=1, dilation=1, kernel_size=3),
            nn.ReLU(True),
            nn.Conv1d(450, 450, stride=1, dilation=1, kernel_size=4),
            nn.ReLU(True),
            nn.Conv1d(450, 450, stride=1, dilation=3, kernel_size=3),
            nn.ReLU(True),
            nn.Conv1d(450, 450, stride=1, dilation=3, kernel_size=3),
            nn.ReLU(True),
            nn.Conv1d(450, 450, stride=1, dilation=3, kernel_size=3),
            nn.ReLU(True),
            nn.Conv1d(450, 450, stride=1, dilation=3, kernel_size=3),
            nn.ReLU(True),
            nn.Conv1d(450, 450, stride=1, dilation=3, kernel_size=3),
            nn.ReLU(True),
            nn.MaxPool1d(3, stride=3),
            st_pool_layer(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(900, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, n_labels),
        )
        self._initialize_weights()

    def load_extractor(self, filename):
        if "tar" in filename:
            state_dict =  torch.load(filename)['state_dict']
        else:
            state_dict = torch.load(filename)

        self.load_state_dict(state_dict)
        # extractor_state_dict = {k:v for k,v in state_dict.items() if
        # 'tdnn' in k}
        # classifier_state_dict = {k:v for k,v in self.state_dict().items() if
        # 'tdnn' not in k}
        # new_state_dict = {**extractor_state_dict, **classifier_state_dict}
        # self.load_state_dict(new_state_dict)
        # assert(len(extractor_state_dict) > 0)
        # print("extractor loaded from {}".format(filename))

    def embed(self, x):
        x = x.squeeze()
        x = x.permute(0,2,1)
        x = self.tdnn(x)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class tdnn_xvector(gTDNN):
    """xvector architecture"""
    def __init__(self, config, n_labels=31):
        super(tdnn_xvector, self).__init__(config, n_labels)
        inDim = config['input_dim']
        self.tdnn = nn.Sequential(
            nn.Conv1d(inDim, 512, stride=1, dilation=1, kernel_size=5),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, stride=1, dilation=3, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, stride=1, dilation=4, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 1500, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(1500),
            nn.ReLU(True),
            st_pool_layer(),
            nn.Linear(3000, 512),
            nn.BatchNorm1d(512),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, n_labels),
        )
        self._initialize_weights()

    def embed(self, x):
        x = x.squeeze(1)
        # (batch, time, freq) -> (batch, freq, time)
        x = x.permute(0,2,1)
        x = self.tdnn(x)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.classifier(x)

        return x

class tdnn_xvector_v1(gTDNN):
    """xvector architecture"""
    def __init__(self, config, n_labels=31):
        super(tdnn_xvector_v1, self).__init__(config, n_labels)
        inDim = config['input_dim']
        self.tdnn = nn.Sequential(
            nn.Conv1d(inDim, 512, stride=1, dilation=1, kernel_size=5),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, stride=1, dilation=3, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, stride=1, dilation=4, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 1500, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(1500),
            nn.ReLU(True),
            st_pool_layer(),
            nn.Linear(3000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, n_labels),
        )
        self._initialize_weights()

    def embed(self, x):
        x = x.squeeze(1)
        # (batch, time, freq) -> (batch, freq, time)
        x = x.permute(0,2,1)
        x = self.tdnn(x)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.classifier(x)

        return x

class tdnn_xvector_untied(nn.Module):
    """xvector architecture
        untying classifier for flexible embedding positon
    """
    def __init__(self, config, base_width=512, n_labels=31):
        super(tdnn_xvector_untied, self).__init__()
        inDim = config['input_dim']
        self.tdnn = nn.Sequential(
            nn.Conv1d(inDim, base_width, stride=1, dilation=1, kernel_size=5),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Conv1d(base_width, base_width, stride=1, dilation=3, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Conv1d(base_width, base_width, stride=1, dilation=4, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Conv1d(base_width, base_width, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Conv1d(base_width, 1500, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(1500),
            nn.ReLU(True),
            st_pool_layer(),
            nn.Linear(3000, base_width),
        )

        last_fc = nn.Linear(base_width, n_labels)

        self.tdnn6_bn = nn.BatchNorm1d(base_width)
        self.tdnn6_relu = nn.ReLU(True)
        self.tdnn7_affine = nn.Linear(base_width, base_width)
        self.tdnn7_bn = nn.BatchNorm1d(base_width)
        self.tdnn7_relu = nn.ReLU(True)
        self.tdnn8_last = last_fc


        self._initialize_weights()

    def embed(self, x):
        x = x.squeeze(1)
        # (batch, time, freq) -> (batch, freq, time)
        x = x.permute(0,2,1)
        x = self.tdnn(x)
        x = self.tdnn6_bn(x)
        x = self.tdnn6_relu(x)
        x = self.tdnn7_affine(x)

        return x

    def forward(self, x):

        x = self.embed(x)
        x = self.tdnn7_bn(x)
        x = self.tdnn7_relu(x)
        x = self.tdnn8_last(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

