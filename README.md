# DFANet
This repo is an unofficial pytorch implementation of DFANet:Deep Feature Aggregation for Real-Time Semantic Segmentation

### Installation

* pytorch==1.0.0
* python==3.6
* numpy
* torchvision
* matplotlib
* opencv-python
* tensorflow
* tensorboardX

### Dataset and pretrained model

Download CityScape dataset and unzip the dataset into `data` folder.Then run the command 'python utils/preprocess.py' to create labels.

### Train the network without pretrained model.
Modify your configuration in `main.py`.

```
run the command  'python main.py'
```

### Segmentation results on CityScape set

![](/image/image.png)

### To do

- [] Train the backbone xceptionA on the ImageNet-1k.

- [] Modify the network and improve the accuracy.

- [] Debug and report the performance.

- [] Schedule the lr

under construction...

### Thanks

