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
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Log images
def log_image(x, opts):
	return tensor2im(x)


def tensor2im(var):
	var = var.detach().transpose((1,2,0)).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))

def vis_faces(log_hooks):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(12, 4 * display_count))
	gs = fig.add_gridspec(display_count, 4)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		vis_faces_with_age(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig

def vis_faces_with_age(hooks_dict, fig, gs, i):
	fig.add_subplot(gs[i, 0])
	plt.imshow(hooks_dict['input_face'])
	plt.title('Input\nOut Sim={:.2f}\nInput Age={:.2f}'.format(float(hooks_dict['diff_input_real']),
	                                                           float(hooks_dict['input_age_real'])))
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target\nIn={:.2f},Out={:.2f}\nTarget Age={:.2f}'.format(float(hooks_dict['diff_views_real']),
	                                                                   float(hooks_dict['diff_target_real']),
	                                                                   float(hooks_dict['target_age_real'])))
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output\nTarget Sim={:.2f}\nOuput Age={:.2f}'.format(float(hooks_dict['diff_target_real']),
	                                                               float(hooks_dict['output_age_real'])))
	fig.add_subplot(gs[i, 3])
	plt.imshow(hooks_dict['recovered_face'])
	plt.title('Recovered\nTarget Sim={:.2f}\nOuput Age={:.2f}'.format(float(hooks_dict['diff_target_cycle']),
																	  float(hooks_dict['output_age_cycle'])))