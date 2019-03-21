
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class st_pool_layer(nn.Module):
    def __init__(self):
        super(st_pool_layer, self).__init__()

    def forward(self, x):
        mean = x.mean(2)
        std = x.std(2)
        stat = torch.cat([mean, std], -1)

        return stat

class tdnn_xvector_base(nn.Module):
    """xvector architecture"""
    def __init__(self):
        super().__init__()
        
    def load_extractor(self, state_dict):
        state_dict.pop("classifier.4.weight")
        state_dict.pop("classifier.4.bias")
        self.load_state_dict(state_dict)

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
                
                
class tdnn_lstm(tdnn_xvector_base):
    """xvector + lstm architecture"""
    def __init__(self, config, base_width, n_labels):
        super().__init__()
        in_dim = config['input_dim']
        self.stat_dim = 1500
        self.tdnn_fr = nn.Sequential(
            nn.Conv1d(in_dim, base_width, stride=1, dilation=1, kernel_size=5),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Conv1d(base_width, base_width, stride=1, dilation=2, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Conv1d(base_width, base_width, stride=1, dilation=3, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Conv1d(base_width, base_width, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Conv1d(base_width, self.stat_dim, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(self.stat_dim),
            nn.ReLU(True),
        )
        
        self.tdnn_uttr = nn.Sequential(
            st_pool_layer(),
            nn.Linear(self.stat_dim*2, base_width//2),
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(base_width//2),
            nn.ReLU(True),
            nn.Linear(base_width//2, base_width),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Linear(base_width, n_labels)
        )
        
        self._initialize_weights()

    def forward(self, x):
        x = x.squeeze(1)
        # (batch, time, freq) -> (batch, freq, time)
        x = x.permute(0,2,1)
        x = self.tdnn_fr(x)
        x = self.tdnn_uttr(x)
        x = self.classifier(x)