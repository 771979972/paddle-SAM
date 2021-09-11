import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import paddle
from PIL import Image
from paddle.io import DataLoader
import sys
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
from argparse import ArgumentParser
# import paddle.distributed as dist

sys.path.append("")
sys.path.append("")

from configs import data_configs
from data.inference_dataset import InferenceDataset
from data.augmentations import AgeTransformer
from utils.common import tensor2im, log_image
# from options.test_options import TestOptions
from models.psp import pSp

# dist.init_parallel_env()
class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to pSp model checkpoint')
		self.parser.add_argument('--data_path', type=str, default='gt_images', help='Path to directory of images to evaluate')
		self.parser.add_argument('--couple_outputs', action='store_true', help='Whether to also save inputs + outputs side-by-side')
		self.parser.add_argument('--resize_outputs', action='store_true', help='Whether to resize outputs to 256x256 or keep at 1024x1024')

		self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')

		# arguments for style-mixing script
		self.parser.add_argument('--n_images', type=int, default=None, help='Number of images to output. If None, run on all data')
		self.parser.add_argument('--n_outputs_to_generate', type=int, default=5, help='Number of outputs to generate per input image.')
		self.parser.add_argument('--mix_alpha', type=float, default=None, help='Alpha value for style-mixing')
		self.parser.add_argument('--latent_mask', type=str, default=None, help='Comma-separated list of latents to perform style-mixing with')

		# arguments for aging
		self.parser.add_argument('--target_age', type=str, default=None, help='Target age for inference. Can be comma-separated list for multiple ages.')

		# arguments for reference guided aging inference
		self.parser.add_argument('--ref_images_paths_file', type=str, default='./ref_images.txt',
                                 help='Path to file containing a list of reference images to use for '
                                      'reference guided inference.')

	def parse(self):
		opts = self.parser.parse_args()
		return opts

def run():
    test_opts = TestOptions().parse()

    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    ckpt = paddle.load(test_opts.checkpoint_path)
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    print(opts)
    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()

    age_transformers = [AgeTransformer(target_age=age) for age in opts.target_age.split(',')]

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_time = []
    for age_transformer in age_transformers:
        print(f"Running on target age: {age_transformer.target_age}")
        global_i = 0
        for input_batch in tqdm(dataloader):
            if global_i >= opts.n_images:
                break
            with paddle.no_grad():
                input_age_batch = [age_transformer(img) for img in input_batch]
                input_age_batch = paddle.stack(input_age_batch)
                input_age_batch = paddle.to_tensor(input_age_batch, paddle.float32)
                tic = time.time()
                result_batch = run_on_batch(input_age_batch, net, opts)
                toc = time.time()
                global_time.append(toc - tic)

                for i in range(len(input_batch)):
                    result = tensor2im(result_batch[i])
                    im_path = dataset.paths[global_i]

                    if opts.couple_outputs or global_i % 100 == 0:
                        input_im = log_image(input_batch[i], opts)
                        resize_amount = (256, 256) if opts.resize_outputs else (1024, 1024)
                        res = np.concatenate([np.array(input_im.resize(resize_amount)),
                                              np.array(result.resize(resize_amount))], axis=1)
                        Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

                    age_out_path_results = os.path.join(out_path_results, age_transformer.target_age)
                    os.makedirs(age_out_path_results, exist_ok=True)
                    im_save_path = os.path.join(age_out_path_results, os.path.basename(im_path))
                    Image.fromarray(np.array(result.resize(resize_amount))).save(im_save_path)
                    global_i += 1

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)


def run_on_batch(inputs, net, opts):
    result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
    return result_batch


if __name__ == '__main__':
    run()
