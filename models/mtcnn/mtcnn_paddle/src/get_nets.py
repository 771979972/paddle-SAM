# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from collections import OrderedDict
import numpy as np

from configs.paths_config import model_paths
PNET_PATH = model_paths["mtcnn_pnet"]
ONET_PATH = model_paths["mtcnn_onet"]
RNET_PATH = model_paths["mtcnn_rnet"]


class Flatten(nn.Layer):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, c, h, w].
        Returns:
            a float tensor with shape [batch_size, c*h*w].
        """

        # without this pretrained model isn't working
        x = x.transpose((0, 1, 3, 2))

        return x.reshape((x.shape[0], -1))


class PNet(nn.Layer):

    def __init__(self):
        super().__init__()

        # suppose we have input with size HxW, then
        # after first layer: H - 2,
        # after pool: ceil((H - 2)/2),
        # after second conv: ceil((H - 2)/2) - 2,
        # after last conv: ceil((H - 2)/2) - 4,
        # and the same for W

        self.features = nn.Sequential(
            ('conv1', nn.Conv2D(3, 10, 3, 1)),
            ('prelu1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2D(2, 2, ceil_mode=True)),

            ('conv2', nn.Conv2D(10, 16, 3, 1)),
            ('prelu2', nn.PReLU(16)),

            ('conv3', nn.Conv2D(16, 32, 3, 1)),
            ('prelu3', nn.PReLU(32))
        )

        self.conv4_1 = nn.Conv2D(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2D(32, 4, 1, 1)

        weights = np.load(PNET_PATH, allow_pickle=True)[()]
        for n, p in self.named_parameters():
            if '_w' in n:
                weights[n] = weights[n.replace('_w','w')]
                del weights[n.replace('_w','w')]
        self.set_state_dict(weights)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4, h', w'].
            a: a float tensor with shape [batch_size, 2, h', w'].
        """
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        a = F.softmax(a, -1)
        return b, a


class RNet(nn.Layer):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            ('conv1', nn.Conv2D(3, 28, 3, 1)),
            ('prelu1', nn.PReLU(28)),
            ('pool1', nn.MaxPool2D(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2D(28, 48, 3, 1)),
            ('prelu2', nn.PReLU(48)),
            ('pool2', nn.MaxPool2D(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2D(48, 64, 2, 1)),
            ('prelu3', nn.PReLU(64)),

            ('flatten', Flatten()),
            ('conv4', nn.Linear(576, 128)),
            ('prelu4', nn.PReLU(128))
        )

        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)

        weights = np.load(RNET_PATH, allow_pickle=True)[()]
        fc = ['features.conv4.weight', 'conv5_1.weight', 'conv5_2.weight']
        for n, p in self.named_parameters():
            if '_w' in n:
                weights[n] = weights[n.replace('_w', 'w')]
                del weights[n.replace('_w', 'w')]
            if n in fc:
                weights[n] = weights[n].transpose()
        self.set_state_dict(weights)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv5_1(x)
        b = self.conv5_2(x)
        a = F.softmax(a, -1)
        return b, a


class ONet(nn.Layer):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            ('conv1', nn.Conv2D(3, 32, 3, 1)),
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2D(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2D(32, 64, 3, 1)),
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2D(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2D(64, 64, 3, 1)),
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2D(2, 2, ceil_mode=True)),

            ('conv4', nn.Conv2D(64, 128, 2, 1)),
            ('prelu4', nn.PReLU(128)),

            ('flatten', Flatten()),
            ('conv5', nn.Linear(1152, 256)),
            ('drop5', nn.Dropout(0.25)),
            ('prelu5', nn.PReLU(256)),
        )

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)

        weights = np.load(ONET_PATH, allow_pickle=True)[()]
        fc = ['features.conv5.weight', 'conv6_1.weight', 'conv6_2.weight', 'conv6_3.weight']
        for n, p in self.named_parameters():
            if '_w' in n:
                weights[n] = weights[n.replace('_w', 'w')]
                del weights[n.replace('_w', 'w')]
            if n in fc:
                weights[n] = weights[n].transpose()
        self.set_state_dict(weights)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        c = self.conv6_3(x)
        a = F.softmax(a, -1)
        return c, b, a
