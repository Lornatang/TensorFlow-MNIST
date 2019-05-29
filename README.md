# TensorFlow-MNIST

This is a classification of the MNIST dataset for TensorFlow v2.0, 
which includes not only the common MNIST dataset, 
but also KMNIST(Japanese dataset), EMNIST(advanced MNIST), 
FASHION_MNIST(fashion dataset), and so on. 
In the model, we defined many models, 
but we all used lenet-5 neural network only in these data sets. 
Because I think that's enough. And Top1 accuracy has reached 99.8%.

## The preparatory work

*requirement*
- `pip install requirements.txt`

You will install all the packages needed for this project.

*machine requirement*
- GPU: A TiTAN V or later.
- Disk: 128G SSD.
- Python version: python3.5 or later.
- CUDA: cuda10.
- CUDNN: cudnn7.4.5 or later.
- Tensorflow-gpu: 2.0.0-alpla0.

## DataSet introduction

### MNIST

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. 
It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.
It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

Four files are available on this site:

[train-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz):  training set images (9912422 bytes) 

[train-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz):  training set labels (28881 bytes) 

[t10k-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-labels-idx3-ubyte.gz):   test set images (1648877 bytes) 

[t10k-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz):   test set labels (4542 bytes)

### KMNIST

Adapted from Kuzushiji Dataset, KMNIST dataset is a drop-in replacement for MNIST dataset. If your software can read the MNIST dataset, it is easy to test the KMNIST dataset by changing the setting. We provide three types of datasets, namely Kuzushiji-MNIST、Kuzushiji-49、Kuzushiji-Kanji, for different purposes.

KMNIST Dataset is created by ROIS-DS Center for Open Data in the Humanities (CODH), based on Kuzushiji Dataset created by National Institute of Japanese Literature. Please refer to the license.

GitHub: **[Repository for Kuzushiji-MNIST, Kuzushiji-49, and Kuzushiji-Kanji](https://github.com/rois-codh/kmnist)**

![The 10 classes of Kuzushiji-MNIST, with the first column showing each character's modern hiragana counterpart.](https://github.com/Lornatang/TensorFlow-MNIST/blob/master/img/introduction_kmnist.png)

Information about Kuzushiji research is available in 2nd CODH Seminar: Kuzushiji Challenge - Future of Machine Recognition and Human Transcription and Kuzushiji Challenge!. Moreover, Kaggle has many examples about how dataset can be used.

Kaggle: **[Kuzushiji-MNIST | Kaggle](https://www.kaggle.com/anokas/kuzushiji/)**

### EMNIST

The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19  and converted to a 28x28 pixel image format and dataset structure that directly matches the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) . Further information on the dataset contents and conversion process can be found in the paper available at https://arxiv.org/abs/1702.05373v1.

#### Formats
The dataset is provided in two file formats. Both versions of the dataset contain identical information, and are provided entirely for the sake of convenience. The first dataset is provided in a Matlab format that is accessible through both Matlab and Python (using the scipy.io.loadmat function). The second version of the dataset is provided in the same binary format as the original MNIST dataset as outlined in [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

#### Dataset Summary
There are six different splits provided in this dataset. A short summary of the dataset is provided below:

- EMNIST ByClass: 814,255 characters. 62 unbalanced classes.

- EMNIST ByMerge: 814,255 characters. 47 unbalanced classes.

- EMNIST Balanced:  131,600 characters. 47 balanced classes.

- EMNIST Letters: 145,600 characters. 26 balanced classes.

- EMNIST Digits: 280,000 characters. 10 balanced classes.

- EMNIST MNIST: 70,000 characters. 10 balanced classes.

The full complement of the NIST Special Database 19 is available in the ByClass and ByMerge splits. The EMNIST Balanced dataset contains a set of characters with an equal number of samples per class. The EMNIST Letters dataset merges a balanced set of the uppercase and lowercase letters into a single 26-class task. The EMNIST Digits and EMNIST MNIST dataset provide balanced handwritten digit datasets directly compatible with the original MNIST dataset.

Please refer to the EMNIST paper [[PDF](http://cn.arxiv.org/pdf/1702.05373v1.pdf) [BIB](http://biometrics.nist.gov/cs_links/EMNIST/emnist.bib)]for further details of the dataset structure.

## Train
**print help**

`python train.py -h`

**quick start**

`python train.py --name lenet --dir training_checkpoint`

You will quickly train a model that classifies the MNIST dataset.

option: 
- *use other dataset*

`python train.py --dataset emnist --classes 62 --name lenet --dir training_checkpoint`

- *use special epochs*

`python train.py --epochs 10 --dataset emnist --classes 62 --name lenet --dir training_checkpoint`

## Prediction

### MNIST acc and loss

![MNIST acc and loss](https://github.com/Lornatang/TensorFlow-MNIST/blob/master/img/mnist_acc_loss.png)

### prediction MNIST

`python3 prediction.py --dir kmnist_checkpoint --path ./datasets/mnist/5.png`

![](https://github.com/Lornatang/TensorFlow-MNIST/blob/master/img/pred_mnist.png)

### EMNIST acc and loss

![EMNIST acc and loss](https://github.com/Lornatang/TensorFlow-MNIST/blob/master/img/emnist_acc_loss.png)

### prediction EMNIST

`python3 prediction.py --dir emnist_checkpoint --path ./datasets/emnist/5.png`

![](https://github.com/Lornatang/TensorFlow-MNIST/blob/master/img/pred_emnist.png)

### KMNIST acc and loss

![KMNIST acc and loss](https://github.com/Lornatang/TensorFlow-MNIST/blob/master/img/kmnist_acc_loss.png)

### prediction KMNIST

`python3 prediction.py --dir kmnist_checkpoint --path ./datasets/kmnist/5.png`

![](https://github.com/Lornatang/TensorFlow-MNIST/blob/master/img/pred_kmnist.png)

## LINCENSE
[Apache License 2.0](https://github.com/Lornatang/TensorFlow-MNIST/blob/master/LICENSE)
