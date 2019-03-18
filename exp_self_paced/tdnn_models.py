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
                
                
class tdnn_xvector(tdnn_xvector_base):
    """xvector architecture"""
    def __init__(self, config, base_width, n_labels):
        super().__init__()
        in_dim = config['input_dim']
        self.stat_dim = 1500
        self.tdnn_fr = nn.Sequential(
            nn.Conv1d(in_dim, base_width, stride=1, dilation=1, kernel_size=5),
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
            nn.Conv1d(base_width, self.stat_dim, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(self.stat_dim),
            nn.ReLU(True),
        )
        
        self.tdnn_uttr = nn.Sequential(
            st_pool_layer(),
            nn.Linear(self.stat_dim*2, base_width),
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Linear(base_width, base_width),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Linear(base_width, n_labels)
        )
        
        self._initialize_weights()

    def load_extractor(self, state_dict):
        state_dict.pop("classifier.4.weight")
        state_dict.pop("classifier.4.bias")
        self.load_state_dict(state_dict)
        
    def fr_feat(self, x):
        x = x.squeeze(1)
        # (batch, time, freq) -> (batch, freq, time)
        x = x.permute(0,2,1)
        x = self.tdnn_fr(x)
        
        return x

    def xvector(self, x):
        x = x.squeeze(1)
        # (batch, time, freq) -> (batch, freq, time)
        x = x.permute(0,2,1)
        x = self.tdnn_fr(x)
        x = self.tdnn_uttr(x)

        return x
    
    def class_clients(self, classes):
        return self.classifier[5].weight[classes]
    
    def embed_logit(self, x):
        x = x.squeeze(1)
        # (batch, time, freq) -> (batch, freq, time)
        x = x.permute(0,2,1)
        x = self.tdnn_fr(x)
        x = self.tdnn_uttr(x)
        embed = self.classifier[:-1](x)
        logit = self.classifier[-1](embed)

        return embed, logit
    
    def forward(self, x):
        x = x.squeeze(1)
        # (batch, time, freq) -> (batch, freq, time)
        x = x.permute(0,2,1)
        x = self.tdnn_fr(x)
        x = self.tdnn_uttr(x)
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

### Self-attention                
class tdnn_xvector_self_att_v0(tdnn_xvector):
    """
        naive self-attention module
    """
    def __init__(self, config, base_width, n_labels):
        self.stat_dim = 1024
        super().__init__(config, base_width, n_labels)
        in_dim = config['input_dim']
        self.tdnn_fr = nn.Sequential(
            nn.Conv1d(in_dim, base_width, stride=1, dilation=1, kernel_size=5),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            # TDNN1 = 2
            nn.Conv1d(base_width, base_width, stride=1, dilation=3, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            # TDNN2 = 5
            nn.Conv1d(base_width, base_width, stride=1, dilation=4, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            # TDNN3 = 8
            nn.Conv1d(base_width, base_width, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            # TDNN4 = 11
            nn.Conv1d(base_width, self.stat_dim, stride=8, dilation=1, kernel_size=10),
        )
        
        self.w_s1 = nn.Parameter(torch.randn(300, self.stat_dim))
        self.w_s2 = nn.Parameter(torch.randn(300))
        
        self.tdnn_uttr = nn.Sequential(
            nn.BatchNorm1d(self.stat_dim),
            nn.ReLU(True),
            st_pool_layer(),
            # ST_pool = 1
            nn.Linear(self.stat_dim*2, base_width),
            nn.BatchNorm1d(base_width),
#             nn.Dropout(),
            nn.ReLU(True),
            # xvector = 4
        )

        self._initialize_weights()
        
    def fr_feat(self, x):
        x = x.squeeze(1)
        # (batch, time, freq) -> (batch, freq, time)
        x = x.permute(0,2,1)
        x = self.tdnn_fr(x)
        
        return x
        
    def att_map(self, x):
        x = x.squeeze(1)
        # (batch, time, freq) -> (batch, freq, time)
        x = x.permute(0,2,1)
        x = self.tdnn_fr(x)
        
#         import ipdb
#         ipdb.set_trace()
        a = torch.matmul(self.w_s1, x)
        a = torch.tanh(a)
        a = torch.matmul(self.w_s2, a)
        a = torch.softmax(a, dim=1)
#         a = F.threshold(a, 0.01, 0)
        
        return a
        
    def forward(self, x):
        x = x.squeeze(1)
        # (batch, time, freq) -> (batch, freq, time)
        x = x.permute(0,2,1)
        x = self.tdnn_fr(x)
        
#         import ipdb
#         ipdb.set_trace()
        a = torch.matmul(self.w_s1, x)
        a = torch.tanh(a)
        a = torch.matmul(self.w_s2, a)
        a = torch.softmax(a, dim=1)
        x = x * a.unsqueeze(1)
        
        x = self.tdnn_uttr(x)
        x = self.classifier(x)

        return x
        
class tdnn_xvector_se(tdnn_xvector_base):
    """
        SE style
    """
    def __init__(self, config, base_width, n_labels):
        super().__init__()
        self.stat_dim = 1500
        in_dim = config['input_dim']
        
        self.tdnn1 = nn.Sequential(
            nn.Conv1d(in_dim, base_width, stride=1, dilation=1, kernel_size=5),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True)
        )
        self.se1 = nn.Sequential(
            nn.Linear(base_width, base_width//4),
            nn.BatchNorm1d(base_width//4),
            nn.ReLU(True),
            nn.Linear(base_width//4, base_width),
            nn.BatchNorm1d(base_width),
            nn.Sigmoid()
        )
        
        # TDNN1 = 2
        self.tdnn2 = nn.Sequential(
            nn.Conv1d(base_width, base_width, stride=1, dilation=3, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
        )
        self.se2 = nn.Sequential(
            nn.Linear(base_width, base_width//4),
            nn.BatchNorm1d(base_width//4),
            nn.ReLU(True),
            nn.Linear(base_width//4, base_width),
            nn.BatchNorm1d(base_width),
            nn.Sigmoid()
        )
        
        # TDNN2 = 5
        self.tdnn3 = nn.Sequential(
            nn.Conv1d(base_width, base_width, stride=1, dilation=4, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
        )
        self.se3 = nn.Sequential(
            nn.Linear(base_width, base_width//4),
            nn.BatchNorm1d(base_width//4),
            nn.ReLU(True),
            nn.Linear(base_width//4, base_width),
            nn.BatchNorm1d(base_width),
            nn.Sigmoid()
        )
        
        # TDNN3 = 8
        self.tdnn4 = nn.Sequential(
            nn.Conv1d(base_width, base_width, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
        )
        self.se4 = nn.Sequential(
            nn.Linear(base_width, base_width//4),
            nn.BatchNorm1d(base_width//4),
            nn.ReLU(True),
            nn.Linear(base_width//4, base_width),
            nn.BatchNorm1d(base_width),
            nn.Sigmoid()
        )
        
        # TDNN3 = 8
        self.tdnn5 = nn.Sequential(
            nn.Conv1d(base_width, self.stat_dim, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(self.stat_dim),
            nn.ReLU(True),
        )
        self.se5 = nn.Sequential(
            nn.Linear(self.stat_dim, base_width),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Linear(base_width, self.stat_dim),
            nn.BatchNorm1d(self.stat_dim),
            nn.Sigmoid()
        )
        
        self.tdnn6 = nn.Sequential(
            nn.Linear(self.stat_dim, base_width),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
        )
        self.se6 = nn.Sequential(
            nn.Linear(base_width, base_width//4),
            nn.BatchNorm1d(base_width//4),
            nn.ReLU(True),
            nn.Linear(base_width//4, base_width),
            nn.BatchNorm1d(base_width),
            nn.Sigmoid()
        )
        
        self.tdnn7 = nn.Sequential(
            nn.Linear(base_width, base_width),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
        )
        self.se7 = nn.Sequential(
            nn.Linear(base_width, base_width//4),
            nn.BatchNorm1d(base_width//4),
            nn.ReLU(True),
            nn.Linear(base_width//4, base_width),
            nn.BatchNorm1d(base_width),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(base_width, n_labels)
        )
        
        self.stat = st_pool_layer()
        
    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0,2,1)
        
#         import ipdb
#         ipdb.set_trace()
        x = self.tdnn1(x)
        x_ = x.mean(-1)
        x_ = self.se1(x_)
        x = x * x_.unsqueeze(-1)
        
        x = self.tdnn2(x)
        x_ = x.mean(-1)
        x_ = self.se2(x_)
        x = x * x_.unsqueeze(-1)
        
        x = self.tdnn3(x)
        x_ = x.mean(-1)
        x_ = self.se3(x_)
        x = x * x_.unsqueeze(-1)

        x = self.tdnn4(x)
        x_ = x.mean(-1)
        x_ = self.se4(x_)
        x = x * x_.unsqueeze(-1)
        
        x = self.tdnn5(x)
        x_ = x.mean(-1)
        x_ = self.se5(x_)
        x = x * x_.unsqueeze(-1)
        
#         x = self.stat(x)
        x = x.mean(-1)
        
        x = self.tdnn6(x)
        x_ = self.se6(x)
        x = x * x_
        
        x = self.tdnn7(x)
        x_ = self.se7(x)
        x = x * x_
        
        x = self.classifier(x)
        
        return x
        
### deep or residual TDNNs
class tdnn_xvector_deep_v0(tdnn_xvector):
    """
        More TDNN layers (current +2 layers)
    """
    def __init__(self, config, base_width, n_labels):
        self.stat_dim = 1500
        super().__init__(config, base_width, n_labels)
        in_dim = config['input_dim']
        self.tdnn_fr = nn.Sequential(
            nn.Conv1d(in_dim, base_width, stride=1, dilation=1, kernel_size=5),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            
            nn.Conv1d(base_width, base_width, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            
            nn.Conv1d(base_width, base_width, stride=1, dilation=3, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            
            nn.Conv1d(base_width, base_width, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            
            nn.Conv1d(base_width, base_width, stride=1, dilation=4, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            
            nn.Conv1d(base_width, base_width, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            
            nn.Conv1d(base_width, base_width, stride=1, dilation=5, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            
            nn.Conv1d(base_width, base_width, stride=1, dilation=1, kernel_size=1),
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
            nn.Linear(self.stat_dim*2, base_width),
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Linear(base_width, base_width),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Linear(base_width, n_labels)
        )
        
        self._initialize_weights()
        
    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0,2,1)
        x = self.tdnn_fr(x)
        x = self.tdnn_uttr(x)
        x = self.classifier(x)

        return x
    
class tdnn_xvector_res_v0(tdnn_xvector_base):
    """
        skip connection
    """
    def __init__(self, config, base_width, n_labels):
        self.stat_dim = 1500
        super().__init__()
        in_dim = config['input_dim']
        self.tdnn_fr1 = nn.Sequential(
            nn.Conv1d(in_dim, base_width, stride=1, dilation=1, kernel_size=5),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
        )
        
        self.tdnn_fr2 = nn.Sequential(
            nn.Conv1d(base_width, base_width, stride=1, dilation=3, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Conv1d(base_width, base_width, stride=1, dilation=3, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReflectionPad1d(6)
        )
        
        self.tdnn_fr3 = nn.Sequential( 
            nn.Conv1d(base_width, base_width, stride=1, dilation=3, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Conv1d(base_width, base_width, stride=1, dilation=3, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReflectionPad1d(6)
        )
        
        self.connect_conv = nn.Conv1d(base_width, self.stat_dim ,kernel_size=1)
        
        self.tdnn_fr4 = nn.Sequential(
            nn.Conv1d(base_width, base_width, stride=1, dilation=3, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Conv1d(base_width, self.stat_dim, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(self.stat_dim),
            nn.ReflectionPad1d(3)
        )
        
        self.tdnn_uttr = nn.Sequential(
            st_pool_layer(),
            # ST_pool = 1
            nn.Linear(self.stat_dim*2, base_width),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            # xvector = 4
        )

        self.classifier = nn.Sequential(
            nn.Linear(base_width, base_width),
            nn.BatchNorm1d(base_width),
            nn.Linear(base_width, n_labels)
        )
        
        self._initialize_weights()
        
    def fr_feat(self, x):
        x = x.squeeze(1)
        x = x.permute(0,2,1)
        x = self.tdnn_fr1(x)
        x = x + self.tdnn_fr2(x)
        x = x + self.tdnn_fr3(x)
        x = self.connect_conv(x) + self.tdnn_fr4(x)
        
        return x
        
    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0,2,1)
        x = self.tdnn_fr1(x)
        x = x + self.tdnn_fr2(x)
        x = x + self.tdnn_fr3(x)
        x = self.connect_conv(x) + self.tdnn_fr4(x)
        x = self.tdnn_uttr(x)
        x = self.classifier(x)

        return x
        
class tdnn_xvector_wide_v0(tdnn_xvector):
    def __init__(self, config, base_width, n_labels):
        self.stat_dim = 1500
        super().__init__(config, base_width, n_labels)
        in_dim = config['input_dim']
        self.tdnn_fr = nn.Sequential(
            nn.Conv1d(in_dim, base_width, stride=1, dilation=1, kernel_size=5),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            # TDNN1 = 2
            nn.Conv1d(base_width, base_width, stride=1, dilation=3, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            # TDNN2 = 5
            nn.Conv1d(base_width, base_width, stride=1, dilation=4, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            # TDNN3 = 8
            nn.Conv1d(base_width, base_width, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            # TDNN4 = 11
            nn.Conv1d(base_width, self.stat_dim, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(self.stat_dim),
            nn.ReLU(True),
        )
        
        self._initialize_weights()
        
    def forward(self, x):
        x = x.squeeze(1)
        # (batch, time, freq) -> (batch, freq, time)
        x = x.permute(0,2,1)
        x = self.tdnn_fr(x)
        x = self.tdnn_uttr(x)
        x = self.classifier(x)

        return x
        

    

    

