# 论文复现：Only a Matter of Style: Age Transformation Using a Style-Based Regression Model

[English](README.md) | [简体中文](README_cn.md)

- **paddle-SAM**
  - [一、简介](#一简介)
  - [二、复现精度](#二复现精度)
  - [三、数据集](#三数据集)
  - [四、环境依赖](四环境依赖)
  - [五、快速开始](五快速开始)
    - [5.1 训练](#51-训练)
    - [5.2 预测](#52-预测)
  - [六、代码结构与详细说明](#六代码结构与详细说明)
    - [6.1 代码结构](#61-代码结构)
    - [6.2 参数说明](#62-参数说明)
    - [6.3 训练流程](#63-训练流程)
    - [6.4 测试流程](#64-测试流程)
  - [七、实验数据及复现心得](#七实验数据比较及复现心得)
  - [八、模型信息](#八模型信息)

## 一、简介

本文提出了一种图像到图像（image-to-image）的年龄转换方法，该方法促使编码器学习将真实面部图像编码到预先训练的无条件 GAN（例如 StyleGAN）的潜在空间中，该编码器会受到给定的年龄变化的影响。
![0](https://img-blog.csdnimg.cn/7d0f39a07f104054b7b1b8fbf4fb9138.png)
本文将老化过程作为输入年龄和目标年龄之间的回归任务，对生成图像的年龄提供细粒度控制（而不是只能老化到给定的年龄），使用预先训练的年龄回归网络来明确地指导编码器生成与期望年龄相符的潜在代码，编码器的任务是将输入的面部图像直接编码为一系列样式向量，以适应所需的年龄变化。然后将这些样式向量输入StyleGAN，以生成表示所需年龄变换的输出图像。

为了明确指导编码器生成相应的潜在代码，本文使用固定的、预训练的年龄回归网络作为训练过程中的年龄约束。本文方法命名为 SAM——Style-based Age Manipulation。因为年龄转换是通过学习到的中间样式表示来控制的。

最后，本文证明了本方法的端到端性质，加上 StyleGAN 丰富的潜在语义空间，允许进一步编辑生成的图像。定性和定量评估表明，与最先进的方法相比，本文的方法具有优势。


论文链接：[Only a Matter of Style: Age Transformation Using a Style-Based Regression Model](https://paperswithcode.com/paper/only-a-matter-of-style-age-transformation) 

赛场：

- [飞桨论文复现挑战赛（第四期）](https://aistudio.baidu.com/aistudio/competition/detail/106)
- [飞桨论文复现挑战赛（第四期）榜单](https://github.com/PaddlePaddle/Paddle/issues/34526)

## 二、复现精度

参考官方开源的 pytorch 版本代码 [https://github.com/yuval-alaluf/SAM](https://github.com/yuval-alaluf/SAM)，基于 paddlepaddle 深度学习框架，对文献算法进行复现后，本项目生成的年龄转换图像如下表所示。图片从左到右分别是：输入图片，模型生成的 0 岁，10 岁，20 岁，30 岁，40 岁，50 岁，60 岁，70 岁，80 岁，90 岁，100 岁图片，更多结果见 example 文件夹。

>验收标准： CelebA 人眼评估生成的图像（可参考论文中展示的生成图片 Figure 4，6，8）

| 模型    | 图片                                                         |
| ------- | ------------------------------------------------------------ |
| Pytorch | <img src="examples/866_torch.jpg" height = "104" width="1248"/><br/> |
| Paddle  | <img src="examples/866_paddle.jpg" height = "104" width="1248"/><br/> |
| Pytorch | <img src="examples/1287_torch.jpg" height = "104" width="1248"/><br/> |
| Paddle  | <img src="examples/1287_paddle.jpg" height = "104" width="1248"/><br/> |
| Pytorch | <img src="examples/2468_torch.jpg" height = "104" width="1248"/><br/> |
| Paddle  | <img src="examples/2468_paddle.jpg" height = "104" width="1248"/><br/> |

以下是使用Paddle复现的其他结果
<img src="examples/000431_paddle.jpg" height = "104" width="1248"/><br/>
<img src="examples/001092.jpg" height = "104" width="1248"/><br/>
<img src="examples/198650.jpg" height = "104" width="1248"/><br/>

> 目前呈现的结果为运行 24000 步保存的模型的结果，据作者称论文的结果为运行了 60000 步所得。

参数配置如下：

> 详见 `paddle-SAM/options/train_options.py`

| 参数名                    | 设置值         | 说明                                       |
| ------------------------- | -------------- | ------------------------------------------ |
| dataset_type              | ffhq_aging     | 训练集                                     |
| exp_dir                   | `exp/test`     | 运行实验的输出结果保存目录                 |
| workers                   | 0              | 训练时DataLoader的workers参数设置          |
| test_workers              | 0              | 测试时DataLoader的workers参数设置          |
| batch_size                | 6              | 训练的批大小                               |
| test_batch_size           | 6              | 测试的批大小                               |
| start_from_encoded_w_plus | store-true     | start from encoded w plus                  |
| use_weighted_id_loss      | store-true     | use weighted id loss                       |
| id_lambda                 | 0.1            | id 损失的权重                              |
| lpips_lambda              | 0.1            | lpips 感知损失的权重                       |
| lpips_lambda_aging        | 0.1            | lpips 感知损失的权重                       |
| lpips_lambda_crop         | 0.6            | lpips 感知损失的权重                       |
| l2_lambda                 | 0.25           | l2 正则化损失的权重                        |
| l2_lambda_aging           | 0，25          | l2 正则化损失的权重                        |
| l2_lambda_crop            | 1              | l2 正则化损失的权重                        |
| w_norm_lambda             | 0，005         | w_norm 层的权重                            |
| aging_lambda              | 5              | 年龄回归损失的权重                         |
| cycle_lambda              | 1              | 循环一致性损失的权重                       |
| input_nc                  | 4              | 输入图像的通道数                           |
| target_age                | uniform_random | 随机生成年龄进行转换和判别，达到细粒度控制 |


Paddle 版源码：[https://github.com/771979972/paddle-SAM](https://github.com/771979972/paddle-SAM)

## 三、数据集

### 3.1 训练集

本项目使用的训练集是 FFHQ 数据集。该数据集由 52,000 张分辨率为 512×512 的高质量 PNG 人物图像组成，并且在年龄、种族和图像背景方面包含相当大的变化。它还对眼镜、太阳镜、帽子等配件有很好的覆盖。图片是从 Flickr 抓取的，因此继承了该网站的所有偏见，并使用 dlib 进行自动对齐和裁剪。

- 数据集概述：
  - 图像数量：52,000 张高质量 PNG 人物图像
  - 图像大小：512 × 512 分辨率
  - 数据集名称：`FFHQ`

- 数据集链接：[FFHQ](https://github.com/NVlabs/ffhq-dataset)


### 3.2 测试集

本项目使用的测试集是 celeba 数据集。（CelebA）是一个大规模人脸属性数据集，拥有超过 20 万张名人头像。该数据集中的图像包含大量姿势变化和背景噪音以及模糊。

- 数据集概述：
  - 图像数量：202599 张人脸图像
  - 图像大小：178 × 218 分辨率
  - 数据集名称：`img_align_celeba`

- 数据集链接：[CELEBA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

数据集下载后分别将图片数据保存在 `FFHQ\` 和 `CelebA_test\` 目录下。

在 `CelebA_test\` 文件夹下已提供 5 张测试图片。

## 四、环境依赖

- 硬件：
  - x86 cpu（RAM >= 16 GB）
  - NVIDIA GPU（VRAM >= 32 GB）
  - CUDA + cuDNN
- 框架：
  - paddlepaddle-gpu==2.1.2（最新稳定版）
- 其它依赖项：
  - numpy >= 1.19.2
  - scipy==1.4.1
  - matplotlib==3.2.1
  - tqdm==4.46.0
  - opencv-python==4.2.0.34

## 五、快速开始

### 5.1 训练

#### step1: 下载预训练模型

下载训练所需的下列预训练模型，将下载后的模型保存在 `pretrained_models\` 中

| 模型(文件名)                                     | Description                                                  |
| :----------------------------------------------- | :----------------------------------------------------------- |
| pSp Encoder(psp_ffhq_encode)                     | pSp 来自 [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel). |
| FFHQ StyleGAN(stylegan2-ffhq-config-f)           | StyleGAN 在FFHQ上训练，来自 [rosinality](https://github.com/rosinality/stylegan2-pytorch) ，输出1024x1024大小的图片 |
| IR-SE50 Model(model_ir_se50)                     | IR SE 模型，来自 [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) 用于训练中计算ID loss。 |
| CurricularFace Backbone(CurricularFace_Backbone) | 预训练的 CurricularFace model，来自 [HuangYG123](https://github.com/HuangYG123/CurricularFace) 用于Similarity的评估。 |
| VGG Age Classifier(dex_agg_classifier)           | VGG age classifier，来自DEX，在FFHQ-Aging数据集上微调。      |
| AlexNet(alexnet和lin_alex)                       | 用于lpips loss计算。                                         |

链接（百度网盘）：[https://pan.baidu.com/s/1G-Ffs8-y93R0ZlD9mEU6Eg](https://pan.baidu.com/s/1G-Ffs8-y93R0ZlD9mEU6Eg )
提取码：m3nb

link (Google Drive)：[https://drive.google.com/drive/folders/1Bs4MQQ9Pw5E2W1-Y9JA2AkGSWwtDzzis?usp=sharing

](https://drive.google.com/drive/folders/1Bs4MQQ9Pw5E2W1-Y9JA2AkGSWwtDzzis?usp=sharing
)

#### step2: 运行训练

首次训练，先编译 ranger 算子：

```
python scripts/compile_ranger.py
```

然后再运行以下命令开始训练：

```
python scripts/train.py \
--dataset_type=ffhq_aging \
--exp_dir=exp/test \
--workers=0 \
--batch_size=6 \
--test_batch_size=6 \
--test_workers=0 \
--val_interval=2500 \
--save_interval=5000  \
--start_from_encoded_w_plus \
--id_lambda=0.1 \
--lpips_lambda=0.1 \
--lpips_lambda_aging=0.1 \
--lpips_lambda_crop=0.6 \
--l2_lambda=0.25 \
--l2_lambda_aging=0.25 \
--l2_lambda_crop=1 \
--w_norm_lambda=0.005 \
--aging_lambda=5 \
--cycle_lambda=1 \
--input_nc=4 \
--target_age=uniform_random \
--use_weighted_id_loss
```

训练日志保存在 `log/vdlrecords.1629623399.log`，训练过程中的测试结果在  `log/timestamp.txt` 中，在当前目录执行以下命令查看日志：

```
from visualdl.server import app
app.run(logdir="log")
```

### 5.2 预测

#### step1: 下载预训练 SAM 模型

SAM 模型下载地址：

百度网盘链接：[https://pan.baidu.com/s/1G-Ffs8-y93R0ZlD9mEU6Eg](https://pan.baidu.com/s/1G-Ffs8-y93R0ZlD9mEU6Eg )（文件名：sam_ffhq_aging.pdparams）
提取码：m3nb

link (Google Drive)：[https://drive.google.com/drive/folders/1Bs4MQQ9Pw5E2W1-Y9JA2AkGSWwtDzzis?usp=sharing](https://drive.google.com/drive/folders/1Bs4MQQ9Pw5E2W1-Y9JA2AkGSWwtDzzis?usp=sharing)

下载后将模型的参数保存在 `pretrained_models\` 中。

#### step2: 图像生成

运行以下命令生成 `--target_age` 参数指定的年龄段的头像：

```
python scripts/inference_side_by_side.py 
--exp_dir=test_inference 
--checkpoint_path=pretrained_models/sam_ffhq_aging.pdparams 
--data_path=images 
--test_batch_size=4 
--test_workers=0 
--target_age=0,10,20,30,40,50,60,70,80,90,100
```

<img src="examples/000431_paddle.jpg" height = "104" width="1248"/><br/>

## 六、代码结构与详细说明

### 6.1 代码结构

```
├── CelebA_test               # 提供十几个用于测试的 celeba 图像
├── configs                   # 数据集路径设置、数据转换定义
├── criteria                  # id_loss、aging_loss、lpip_loss 等损失函数的定义
├── datasets                  # Dataset 封装数据集
├── examples                  # 十几个测试生成的年龄转换图像
├── log                       # 训练日志文件
├── models                    # SAM 模型定义
│   ├── encoders              # pSp 编码器定义
│   ├── stylegan2             # stylegan2模型定义
│   ├── dex_agg.py            # 使用dex数据集训练的用于年龄判别的vgg
│   ├── psp.py                # pSp 编码器定义
├── options                   # 训练及测试的参数设置
├── scripts                   # 训练、测试和 ranger 算子的运行脚本
├── training                  # 训练策略定义 coach_aging.py 和 ranger 算子定义 
├── utils                     # 工具包
├── README.md                 # 主页 readme
```

### 6.2 参数说明

见 [二、复现精度](#二复现精度)

### 6.3 训练流程

见 [五、快速开始](#五快速开始)

### 6.4 测试流程

见 [五、快速开始](#五快速开始)

## 七、实验数据比较及复现心得

复现 SAM 的过程是充满惊喜的，虽然其代码量乍一看让人一乍 :scream:，但是大部分的代码都能借鉴飞浆的开源项目 [paddleGAN](https://github.com/PaddlePaddle/PaddleGAN)，其它代码似乎只需要照着 API 映射表更改即可。解决困难的过程中我们有一些收获：

1. SAM 需要借助好几个预训练模型来构建损失函数，这些模型的转换需要考虑到 paddle 与 pytorch 的 state_dict 的键值的差异，以及全连接层需要转置的问题；
2. 使用转换好的预训练 SAM 模型训练 paddle 版的 SAM 时，最初生成的人脸都是同一个人，我猜大概是预训练的平均隐藏编码自带的信息导致的；
3. 第一次训练到生成的人脸都已经很清晰之后，我们发现竟然没有年龄转换的效果，于是定位问题在 aging_loss 损失函数上，然后发现 dex_vgg.py 中定义的 VGG_Net 在 forward 过程中使用字典来保存中间层的结果，这在 paddle 版本实现下导致了 aging_loss 的梯度无法反向传播，直接把中间结果改成 x 即可解决问题。
4. ranger 算子的重写也是非常关键的一环。
5. 非常感谢原作者 @yuval-alaluf 的帮助，我们在官方源码提的 [issue](https://github.com/yuval-alaluf/SAM/issues/32) 得到了作者耐心、及时的指导。致敬开源精神！:crown::crown::crown:

## 八、模型信息

| 信息             | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| 发布者           | 不想改代码                                                   |
| 时间             | 2021.08                                                      |
| 框架版本         | paddlepaddle==2.1.2（最新稳定版）                            |
| 应用场景         | GAN 图像生成                                                 |
| 支持硬件         | GPU、CPU（RAM >= 16 GB）                                     |
| CELEBA数据集下载 | [FFHQ](https://github.com/NVlabs/ffhq-dataset)、[CELEBA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| AI Studio 地址   | [https://aistudio.baidu.com/aistudio/](https://aistudio.baidu.com/aistudio/) |
| Github 地址      | [paddle-SAM](https://github.com/771979972/paddle-SAM)        |

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
