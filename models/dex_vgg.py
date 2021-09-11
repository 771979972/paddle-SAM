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
import paddle.nn as nn
import paddle.nn.functional as F

class VGG(nn.Layer):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()

        # vgg modules
        self.conv1_1 = nn.Conv2D(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2D(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2D(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2D(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2D(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2D(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2D(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2D(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2D(512, 512, kernel_size=3, padding=1)

        self.fc6 = nn.Linear(25088, 4096, bias_attr=True)
        self.fc7 = nn.Linear(4096, 4096, bias_attr=True)
        self.fc8_101 = nn.Linear(4096, 101, bias_attr=True)
        if pool == 'max':
            self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2D(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2D(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2D(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2D(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2D(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2D(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2D(kernel_size=2, stride=2)
    
    def forward(self, x):
        
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool5(x)
        x = x.reshape((x.shape[0], -1))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8_101(x)
        return x