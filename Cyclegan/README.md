# CycleGan-jittor

Inplementation of [CycleGan](https://arxiv.org/pdf/1703.10593.pdf) in jittor, largely based on and model aligned with [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

To see more informations , check docs folder.

## Prerequisites

* Linux or macOS
* Python 3
* CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation

Install [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/download/) and [see how to use it](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html).

* For pip users, please type the command `pip install -r requirements.txt`.
* For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.

### Prepare Datasets

To see more about datasets, please check the [datasets](https://github.com/zhouwy19/XNN-Project/blob/main/Cyclegan/docs/datasets.md).

Download a dataset(e.g. dacades)：

```
bash ./datasets/download_pix2pix_dataset.sh facades
```

### Train the Model

To see more information for train and test, check the [tips](https://github.com/zhouwy19/XNN-Project/blob/main/Cyclegan/docs/tips.md).

Train a model:

```
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```

To see more intermediate results, check out ./checkpoints/maps_cyclegan/web/index.html.

### Test the Model

Test a model:

```
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```

The test results will be saved to a html file ` ./results/maps_cyclegan/latest_test/index.html.`

### Evaluations

to be updated
