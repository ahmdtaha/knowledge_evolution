# knowledge_evolution
PyTorch implementation of the Knowledge Evolution training approach and Split-Nets

### TL;DR
We subclass neural layers and define the mask inside the subclass. When creating a new network, we simply use our SplitConv and SplitLinear instead of the standard nn.Conv2d and nn.Linear.

## Requirements

* Python 3+ [Tested on 3.7]
* PyTorch 1.X [Tested on torch 1.6.0 and torchvision 0.6.0]

[//]: # "## ImageNet Pretrained Models"



## Usage example

The Flower102Pytorch loader (`data>flower.py`) works directly with this [Flower102 dataset](https://drive.google.com/file/d/1a0h-bPfh3EAnVXBg2Yr1UCPBcfisdEoe/view?usp=sharing). This is the original flower102, but with an extra `list` directory that contains csv files for trn, val and tst splits. Feel free to download the flower dataset from [oxford website](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), just update `data>flower.py` accordingly. 

The following table shows knowledge evolution in both the dense (even rows) and slim (odd rows) using Flower102 on ResNet18.
As the number of generation increases, both the dense and slim networks' performance increases.

![Our implementation performance](./imgs/dense_slim.jpg)  

### TODO LIST
* Disable CS_KD
* Document the important flags
* Add sample results to README file
* Add GoogleNet support

Contributor list
----------------
1. [Ahmed Taha](http://www.cs.umd.edu/~ahmdtaha/)

I want to give credit to [Vivek Ramanujan and Mitchell Wortsman's repos](https://github.com/allenai/hidden-networks). My implementation uses a lot of utils and functions from their code

### MISC Notes
* This repository delivers a knowledge evolution implementation in its simplest form. Accordingly, I disabled CS_KD baseline because it requires a specific sampling implementation. 
## Release History
* 1.0.0
    * First code commit on 10 Dec 2020
    * Submit/Test Split_googlenet code commit on 19 Dec 2020
    * Submit/Test Split_densenet code commit on 20 Dec 2020
