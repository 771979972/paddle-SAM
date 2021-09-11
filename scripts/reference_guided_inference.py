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
from argparse import Namespace
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import paddle
from paddle.io import DataLoader

import sys
sys.path.append("")
sys.path.append("")

from configs import data_configs
from data.inference_dataset import InferenceDataset
from data.augmentations import AgeTransformer
from utils.common import log_image
# from options.test_options import TestOptions
from models.psp import pSp
from argparse import ArgumentParser

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

    out_path_results = os.path.join(test_opts.exp_dir, 'reference_guided_inference')
    os.makedirs(out_path_results, exist_ok=True)

    # update test options with options used during training
    ckpt = paddle.load(test_opts.checkpoint_path)
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()

    age_transformers = [AgeTransformer(target_age=age) for age in opts.target_age.split(',')]

    print(f'Loading dataset for {opts.dataset_type}')
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()

    source_dataset = InferenceDataset(root=opts.data_path,
                                      transform=transforms_dict['transform_inference'],
                                      opts=opts)
    source_dataloader = DataLoader(source_dataset,
                                   batch_size=opts.test_batch_size,
                                   shuffle=False,
                                   num_workers=int(opts.test_workers),
                                   drop_last=False)

    ref_dataset = InferenceDataset(paths_list=opts.ref_images_paths_file,
                                   transform=transforms_dict['transform_inference'],
                                   opts=opts)
    ref_dataloader = DataLoader(ref_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1,
                                drop_last=False)

    if opts.n_images is None:
        opts.n_images = len(source_dataset)

    for age_transformer in age_transformers:
        target_age = age_transformer.target_age
        print(f"Running on target age: {target_age}")
        age_save_path = os.path.join(out_path_results, str(target_age))
        os.makedirs(age_save_path, exist_ok=True)
        global_i = 0
        for i, source_batch in enumerate(tqdm(source_dataloader)):
            if global_i >= opts.n_images:
                break
            results_per_source = {idx: [] for idx in range(len(source_batch))}
            with paddle.no_grad():
                for ref_batch in ref_dataloader:
                    source_batch = paddle.to_tensor(source_batch, paddle.float32)
                    ref_batch = paddle.to_tensor(ref_batch, paddle.float32)
                    source_input_age_batch = [age_transformer(img) for img in source_batch]
                    source_input_age_batch = paddle.stack(source_input_age_batch)

                    # compute w+ of ref images to be injected for style-mixing
                    ref_latents = net.pretrained_encoder(ref_batch) + net.latent_avg

                    # run age transformation on source images with style-mixing
                    res_batch_mixed = run_on_batch(source_input_age_batch, net, opts, latent_to_inject=ref_latents)

                    # store results
                    for idx in range(len(source_batch)):
                        results_per_source[idx].append([ref_batch[0], res_batch_mixed[idx]])

                # save results
                resize_amount = (256, 256) if opts.resize_outputs else (1024, 1024)
                for image_idx, image_results in results_per_source.items():
                    input_im_path = source_dataset.paths[global_i]
                    image = source_batch[image_idx]
                    input_image = log_image(image, opts)
                    # initialize results image
                    ref_inputs = np.zeros_like(input_image.resize(resize_amount))
                    mixing_results = np.array(input_image.resize(resize_amount))
                    for ref_idx in range(len(image_results)):
                        ref_input, mixing_result = image_results[ref_idx]
                        ref_input = log_image(ref_input, opts)
                        mixing_result = log_image(mixing_result, opts)
                        # append current results
                        ref_inputs = np.concatenate([ref_inputs,
                                                     np.array(ref_input.resize(resize_amount))],
                                                    axis=1)
                        mixing_results = np.concatenate([mixing_results,
                                                         np.array(mixing_result.resize(resize_amount))],
                                                        axis=1)
                    res = np.concatenate([ref_inputs, mixing_results], axis=0)
                    save_path = os.path.join(age_save_path, os.path.basename(input_im_path))
                    Image.fromarray(res).save(save_path)
                    global_i += 1


def run_on_batch(inputs, net, opts, latent_to_inject=None):
    if opts.latent_mask is None:
        result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
    else:
        latent_mask = [int(l) for l in opts.latent_mask.split(",")]
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get output image with injected style vector
            res, res_latent = net(paddle.to_tensor(input_image.unsqueeze(0), paddle.float32),
                                  latent_mask=latent_mask,
                                  inject_latent=latent_to_inject,
                                  alpha=opts.mix_alpha,
                                  resize=opts.resize_outputs,
                                  return_latents=True)
            result_batch.append(res)
        result_batch = paddle.concat(result_batch, 0)
    return result_batch


if __name__ == '__main__':
    run()
