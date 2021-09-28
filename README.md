论文复现：Only a Matter of Style: Age Transformation Using a Style-Based Regression Model

*****

[English](README.md)|[简体中文](README_cn.md)

* **paddle-SAM**
  * [1 Introduction](1Introduction)
  * [2 Result](2Result)
  * [3 Dataset](3Dataset)
  * [4 Environment](4Environment)
  * [5 Prentrained models](5Prentrainedmodels)
  * [6 Quick start](6Quickstart)
    * [6.1 Inference](6.1Inference)
    * [6.2 train](6.2train)
    * [6.3 others](6.3others )
  * [7 Code structure](7Codestructure)
    * [7.1 structure](7.1structure)
    * [7.2 Parameter description](7.2 Parameterdescription)
  * [8 Model information](8Modelinformation)

# 

# 1 Introduction

***

This protest reproduces SAM based on paddlepaddle framework.SAM is an image-to-imagetranslation method that learns to directly encode real facial images into the latent space of a pre-trained unconditional GAN subject to a given aging shift.

#### Paper

* [1] Y  Alaluf,  Patashnik O ,  Cohen-Or D . Only a Matter of Style: Age Transformation Using a Style-Based Regression Model[J].  2021.

#### Reference project

* [https://github.com/yuval-alaluf/SAM](https://github.com/yuval-alaluf/SAM)

#### **Project on Ai Studio**

* notebook

[https://aistudio.baidu.com/aistudio/projectdetail/2331297](https://aistudio.baidu.com/aistudio/projectdetail/2331297)

# **2 Result**

* The current presented is the result of the model that is saved 24,000 steps. According to the author's results, the results are running 60000 steps. The picture from left to right is: Enter the picture, the model is 0 years old, 10 years old, 20 years old, 30 Years, 40 years old, 60 years old, 70 years old, 80 years old, 90 years old, 100 years old.

#### **Visual comparison**

|  模型   |                             图片                             |
| :-----: | :----------------------------------------------------------: |
| Pytorch | ![ 1  ](https://ai-studio-static-online.cdn.bcebos.com/f694aa85db1f41b99685aa74984512f7f5ffadd289ab40bbae253b77572e3d44) |
| Paddle  | ![1](https://ai-studio-static-online.cdn.bcebos.com/bbd4c8b5d7624acfa74280f237a2160502e5834063c84f008019d6212351d096) |
| Pytorch | ![1](https://ai-studio-static-online.cdn.bcebos.com/01e35228b4ca451f9f58091a374de6049eb68b4f92bf4b1a8f483317db6f56a3) |
| Paddle  | ![1](https://ai-studio-static-online.cdn.bcebos.com/2a727e5efa5a45aa86cbcd4cd375d5a849dd8d7fff244f7e9fa1e7a65de72dba) |
| Pytorch | ![1](https://ai-studio-static-online.cdn.bcebos.com/47dfbd675ae141e4b9cc10ed8c7b39413ffd9e46ffaa44d0bbc6a21684f1e413) |
| Paddle  | ![1](https://ai-studio-static-online.cdn.bcebos.com/51b2799fdf2e45fba39d4bf2b7f7959d9092cea658824fffa52278449df08646) |

# **3 Datasets**

* Training： [FFHQ-1024](https://github.com/NVlabs/ffhq-dataset).  saved in `SAM/data/FFHQ/`.


- Testing：[CelebA-HQ](https://aistudio.baidu.com/aistudio/datasetdetail/49226).saved in`SAM/data/CelebA_test/`.

# 4 Environment

Hardware：GPU、CPU

Framework：PaddlePaddle >=2.0.0

# 5 Pretrained models

Pretrained models saved in`pretrained_models/`.

| Pretrained models                                         | Description                                                  |
| --------------------------------------------------------- | ------------------------------------------------------------ |
| FFHQ StyleGAN(stylegan2-ffhq-config-f.pdparams)           | StyleGAN trained with the FFHQ dataset from[rosinality](https://github.com/rosinality/stylegan2-pytorch) ，output size:1024x1024 |
| IR-SE50 Model(model_ir_se50.pdparams)                     | IR_SE model ([TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch))trained for computering ID loss. |
| CurricularFace Backbone(CurricularFace_Backbone.paparams) | Pretrained CurricularFace model([HuangYG123](https://github.com/HuangYG123/CurricularFace))evaled Similarity |
| AlexNet(alexnet.pdparams和lin_alex.pdparams)              | computered lpips loss                                        |
| StyleGAN Inversion(psp_ffhq_inverse.pdparams)             | pSp trained with the FFHQ dataset for StyleGAN inversion.    |

Baidu driver：[https://pan.baidu.com/s/1G-Ffs8-y93R0ZlD9mEU6Eg](https://pan.baidu.com/s/1G-Ffs8-y93R0ZlD9mEU6Eg) password：m3nb

# 6 Quick start

    # clone this repo
    git clone [https://github.com/771979972/paddle-SAM.git]
    cd work

#### 6.1 **Inference**

    python SAM/scripts/inference_side_by_side.py 
    --exp_dir=exp/test 
    --checkpoint_path=SAM/pretrained_models/sam_ffhq_aging.pdparams 
    --data_path=SAM/data/CelebA_test 
    --test_batch_size=4 
    --test_workers=0 
    --target_age=0,10,20,30,40,50,60,70,80,90,100

Configuration Environment

    !pip install --upgrade matplotlib
    python SAM/scripts/compile_ranger.py

#### 6.2 Train

    python SAM/scripts/train.py /
    --dataset_type=ffhq_encode /
    --exp_dir=exp/test /
    --workers=0 /
    --batch_size=8/
    --test_batch_size=8 /
    --test_workers=0 /
    --val_interval=2500 /
    --save_interval=5000 /
    --encoder_type=GradualStyleEncoder/
    --start_from_latent_avg /
    --lpips_lambda=0.8 \--l2_lambda=1 /
    --id_lambda=0.1 /
    --optim_name=ranger

#### 6.3 **others**

LPIPS

    python SAM/scripts/calc_losses_on_images.py /
    --mode lpips /
    --data_path=SAM/inference/inference_results /
    --gt_path=SAM/data/CelebA_test/


MSE

    python SAM/scripts/calc_losses_on_images.py /
    --mode l2 /
    --data_path=SAM/inference/inference_results /
    --gt_path=SAM/data/CelebA_test/

Similarity

    python SAM/scripts/calc_id_loss_parallel.py /
    --data_path=SAM/inference/inference_results /
    --gt_path=SAM/data/CelebA_test/

# 7 Code structure

#### 7.1 **Structure**

    ├─config          # 配置
    ├─data            #数据集加载
       ├─CelebA_test  # 测试数据图像
    ├─models          # 模型
        ├─encoders    # 编码器
        ├─loss        # 损失函数
        ├─utils       # 编译算子
    ├─scripts         #算法执行
        trian         #训练
        inference     #测试
        inference_side_by_side    #测试
        reference_guided_inference    #测试
    ├─utils           # 工具代码
    │  README.md      #英文readme
    │  README_cn.md   #中文readme

#### 7.2 **Parameter description**

| Parameter                 | Default        |
| ------------------------- | -------------- |
| config                    | None           |
| dataset_type              | ffhq_aging     |
| exp_dir                   | `exp/test`     |
| workers                   | 0              |
| test_workers              | 0              |
| batch_size                | 6              |
| test_batch_size           | 6              |
| start_from_encoded_w_plus | store-true     |
| use_weighted_id_loss      | store-true     |
| id_lambda                 | 0.1            |
| lpips_lambda              | 0.1            |
| lpips_lambda_aging        | 0.1            |
| lpips_lambda_crop         | 0.6            |
| l2_lambda                 | 0.25           |
| l2_lambda_aging           | 0，25          |
| l2_lambda_crop            | 1              |
| w_norm_lambda             | 0，005         |
| aging_lambda              | 5              |
| cycle_lambda              | 1              |
| input_nc                  | 4              |
| target_age                | uniform_random |

# 8 Model information

The overall information of the model is as follows:

| Information | Descriptions     |
| ----------- | ---------------- |
| Version     | Paddle 2.1.2     |
| Application | Image Generation |
| Hardware    | GPU / CPU        |

## License

```
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
```
