import paddle
from paddle import nn
import paddle.nn.functional as F
from configs.paths_config import model_paths
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
from models.dex_vgg import VGG

class AgingLoss(nn.Layer):

    def __init__(self, opts):
        super(AgingLoss, self).__init__()
        print("Loading Age Classifier")
        self.age_net = VGG()
        state_dict = paddle.load(model_paths['age_predictor'])
        self.age_net.set_state_dict(state_dict)
        self.age_net.eval()

        self.min_age = 0
        self.max_age = 100
        self.age = paddle.arange(self.min_age,self.max_age+1,dtype=paddle.float32)

        self.opts = opts

    
    def __get_predicted_age(self, age_pb):
        predict_age_pb = F.softmax(age_pb)
        predict_age = predict_age_pb.matmul(self.age)
        return predict_age
    
    def extract_ages(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        predict_age_pb = self.age_net(x)
        predicted_age = self.__get_predicted_age(predict_age_pb)
        return predicted_age
    
    def forward(self, y_hat, y, target_ages, id_logs, label=None):
        n_samples = y.shape[0]

        if id_logs is None:
            id_logs = []
        
        input_ages = self.extract_ages(y)/100.
        output_ages = self.extract_ages(y_hat)/100.
    
        for i in range(n_samples):
            if len(id_logs) > i:
                id_logs[i].update({f'input_age_{label}': float(input_ages[i]) * 100,
                                   f'output_age_{label}': float(output_ages[i])*100,
                                   f'target_age_{label}': float(target_ages[i])*100})
            else:
                id_logs.append({f'input_age_{label}': float(input_ages[i]) * 100,
                                f'output_age_{label}': float(output_ages[i]) * 100,
                                f'target_age_{label}': float(target_ages[i]) * 100})
        
        loss = F.mse_loss(output_ages, target_ages)
        return loss, id_logs

