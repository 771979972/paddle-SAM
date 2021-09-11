#encoding=utf8
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
import numpy as np
import paddle

class AgeTransformer(object):
	def __init__(self, target_age):
		self.target_age = target_age
	
	def __call__(self, img):
		img = self.add_aging_channel(img)
		return img
	
	def add_aging_channel(self, img):
		target_age = self.__get_target_age()
		target_age = int(target_age) / 100
		img = paddle.concat((img, target_age*paddle.ones((1, img.shape[1], img.shape[2]))))
		return img
	
	def __get_target_age(self):
		if self.target_age == "uniform_random":
			return np.random.randint(low=0., high=101, size=1)[0]
		else:
			return self.target_age
