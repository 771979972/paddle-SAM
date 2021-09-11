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
from paddle import nn
from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone


class IDLoss(nn.Layer):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.set_state_dict(paddle.load(model_paths['ir_se50']))
        self.face_pool = paddle.nn.AdaptiveAvgPool2D((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, x, label=None, weights=None):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)

        total_loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])
            
            if label is None:
                id_logs.append({'diff_target': float(diff_target),
                                'diff_input': float(diff_input),
                                'diff_views': float(diff_views)})
            else:
                id_logs.append({f'diff_target_{label}': float(diff_target),
                                f'diff_input_{label}': float(diff_input),
                                f'diff_views_{label}': float(diff_views)})

            loss = 1. - diff_target
            if weights is not None:
                loss = float(weights[i]) * loss
            total_loss += loss
            id_diff = float(diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1
            
        return total_loss / float(count), sim_improvement / float(count), id_logs
        # return nn.functional.mse_loss(y_hat,y), sim_improvement / count, id_logs
