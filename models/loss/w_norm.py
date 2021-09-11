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


class WNormLoss(nn.Layer):

	def __init__(self, opts):
		super(WNormLoss, self).__init__()
		self.opts = opts

	def forward(self, latent, latent_avg=None):
		if self.opts.start_from_latent_avg or self.opts.start_from_encoded_w_plus:
			latent = latent - latent_avg
		return paddle.sum(latent.norm(2, (1, 2))) / latent.shape[0]
