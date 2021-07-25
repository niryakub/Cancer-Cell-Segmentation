# Cancer Instance Segmentation and Classification
Detecting &amp; Segmenting cancer cells in histology images from PanNuke dataset. Using Computer Vision + Deep Learning neural networks architectures.
This Project implements the kaggle challenge: [Cancer Instance Segmentation and Classification](https://www.kaggle.com/andrewmvd/cancer-inst-segmentation-and-classification). 
The goal is to segment and classify each cell in the histology images.
PyTorch implementation.

## Dataset
The dataset contains nuclei instance segmentation and classification images with exhaustive nuclei labels across 19 different tissue types. The dataset contains 205,343 labeled nuclei, each with an instance segmentation mask. [An Open Pan-Cancer Histology Dataset for Nuclei Instance Segmentation and Classification](https://link.springer.com/chapter/10.1007/978-3-030-23937-4_2)

## Architectures
We use UNet architecture and MicroNet architecture, related files in model.py.  [official MicroNet paper](https://arxiv-org.ezproxy.haifa.ac.il/abs/1804.08145)

## jupyter notebook
.ipynb google colab notebooks are available, in which you can find the entire process from creating the datasets, the model, the training process, the evaluation process, and vizualization.

## Predictions
### Masks Predictions
![alt text](https://i.ibb.co/xhCYj3z/vis-masks.png)
![alt text](https://i.ibb.co/wYkxpk6/vis-masks-2.png)
![alt text](https://i.ibb.co/tBcc8Y9/vis-masks-3.png)
### Vizualizing Results
![alt text](https://i.ibb.co/SNff1XK/vis-color.png)
![alt text](https://i.ibb.co/znjcMfQ/vis-color-2.png)
