﻿﻿# README_cn
*****
### English|简体中文

* SAM
    * 一、简介
    * 二、复现结果
    * 三、数据集
    * 四、环境依赖
    * 五、预训练模型
    * 六、快速开始
         * inference
         * 训练
         * 计算其他指标
    * 七、代码结构与详细说明
         * 代码结构
         * 参数说明
    * 八、模型信息

# **一、简介**

***
本项目基于paddlepaddle框架复现SAM。SAM算法是一种图像到图像的转换方法，可学习将真实人脸图像直接编码到预训练无条件GAN(如StyleGAN)潜空间中，并受给定年龄变换的影响。

#### **论文**
* [1] Y  Alaluf,  Patashnik O ,  Cohen-Or D . Only a Matter of Style: Age Transformation Using a Style-Based Regression Model[J].  2021.
#### **参考项目**
* [https://github.com/yuval-alaluf/SAM](https://github.com/yuval-alaluf/SAM)
#### **项目aistudio地址**
* notebook任务：[https://aistudio.baidu.com/aistudio/projectdetail/2331297](https://aistudio.baidu.com/aistudio/projectdetail/2331297)

# **二、复现结果**

* 目前呈现的结果为运行24000步保存的模型的结果，据作者称论文的结果为运行了60000步.图片从左到右分别是：输入图片，模型生成的0岁，10岁，20岁，30岁，40岁，50岁，60岁，70岁，80岁，90岁，100岁图片

#### **视觉对比**

|  模型  | 图片   |
|  :----:| :----: |
| Pytorch| ![ 1  ](https://ai-studio-static-online.cdn.bcebos.com/f694aa85db1f41b99685aa74984512f7f5ffadd289ab40bbae253b77572e3d44) |
| Paddle | ![1](https://ai-studio-static-online.cdn.bcebos.com/bbd4c8b5d7624acfa74280f237a2160502e5834063c84f008019d6212351d096) |
| Pytorch| ![1](https://ai-studio-static-online.cdn.bcebos.com/01e35228b4ca451f9f58091a374de6049eb68b4f92bf4b1a8f483317db6f56a3) |
| Paddle | ![1](https://ai-studio-static-online.cdn.bcebos.com/2a727e5efa5a45aa86cbcd4cd375d5a849dd8d7fff244f7e9fa1e7a65de72dba)|
| Pytorch| ![1](https://ai-studio-static-online.cdn.bcebos.com/47dfbd675ae141e4b9cc10ed8c7b39413ffd9e46ffaa44d0bbc6a21684f1e413)|
| Paddle | ![1](https://ai-studio-static-online.cdn.bcebos.com/51b2799fdf2e45fba39d4bf2b7f7959d9092cea658824fffa52278449df08646) |
# **三、数据集**

* 训练集下载： [FFHQ训练集](https://github.com/NVlabs/ffhq-dataset)。图片数据保存在`SAM/data/FFHQ/`。
* 测试集下载：[CelebA](https://github.com/NVlabs/ffhq-dataset)。图片数据保存在`SAM/data/CelebA_test/`。

# **四、环境依赖**

* 硬件：GPU、CPU

框架：PaddlePaddle >=2.0.0

# **五、预训练模型**

* 下载后将模型的参数保存在`work\pretrained_models\`中。
|  模型(文件名)   | Description  |
|  ----  | ----  |
| FFHQ StyleGAN(stylegan2-ffhq-config-f.pdparams) | StyleGAN 在FFHQ上训练，来自 [rosinality](https://github.com/rosinality/stylegan2-pytorch) ，输出1024x1024大小的图片 |
| IR-SE50 Model(model_ir_se50.pdparams)| IR-SE 模型，来自 [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) 用于训练中计算ID loss。 |
| CurricularFace Backbone(CurricularFace_Backbone.paparams)    |     预训练的 CurricularFace model，来自 [HuangYG123](https://github.com/HuangYG123/CurricularFace) 用于Similarity的评估。  |
|   AlexNet(alexnet.pdparams和lin_alex.pdparams)    |   	用于lpips loss计算。    |
| StyleGAN Inversion(psp_ffhq_inverse.pdparams)      |   pSp trained with the FFHQ dataset for StyleGAN inversion.    |

链接：[https://pan.baidu.com/s/1G-Ffs8-y93R0ZlD9mEU6Eg](https://pan.baidu.com/s/1G-Ffs8-y93R0ZlD9mEU6Eg) 提取码：m3nb

# **六、快速开始**

    # clone this repo
    git clone [https://github.com/771979972/paddle-SAM.git]
    cd paddle-SAM
#### **Inference**

    python SAM/scripts/inference_side_by_side.py 
    --exp_dir=exp/test 
    --checkpoint_path=SAM/pretrained_models/sam_ffhq_aging.pdparams 
    --data_path=SAM/data/CelebA_test 
    --test_batch_size=4 
    --test_workers=0 
    --target_age=0,10,20,30,40,50,60,70,80,90,100

#### 首先配置环境

    !pip install --upgrade matplotlib
    python SAM/scripts/compile_ranger.py

#### 然后再训练

    python SAM/scripts/train.py /
    --dataset_type=ffhq_encode /
    --exp_dir=exp/test /
    --workers=0 /
    --batch_size=8 /
    --test_batch_size=8 /
    --test_workers=0 /
    --val_interval=2500/
    --save_interval=5000 /
    --encoder_type=GradualStyleEncoder /
    --start_from_latent_avg /
    --lpips_lambda=0.8 \--l2_lambda=1 /
    --id_lambda=0.1 /
    --optim_name=ranger

#### **计算其他指标**
计算LPIPS

    python SAM/scripts/calc_losses_on_images.py /
    --mode lpips /
    --data_path=SAM/inference/inference_results /
    --gt_path=SAM/data/CelebA_test/


计算MSE

    python SAM/scripts/calc_losses_on_images.py /
    --mode l2 /
    --data_path=SAM/inference/inference_results /
    --gt_path=SAM/data/CelebA_test/

计算Similarity

    python SAM/scripts/calc_id_loss_parallel.py /
    --data_path=SAM/inference/inference_results /
    --gt_path=SAM/data/CelebA_test/

# **七、代码结构与详细说明**

#### **代码结构**

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

#### **参数说明**

|  参数   | 默认值  |
| --  | -- |
| config  | None, 必选 |

# **八、模型信息**

* 模型的总体信息如下：
|信息|说明|
|--|--|
| 框架版本| Paddle 2.1.2 |
| 应用场景|图像生成|
|支持硬件|GPU / CPU|

