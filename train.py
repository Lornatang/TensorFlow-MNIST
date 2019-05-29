# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""main func."""

# Import dataset and model network
from datasets import mnist, kmnist, emnist
import tensorflow as tf

from models import LeNet
from models import AlexNet
from models import VGG16
from models import VGG19

# plt pic
import matplotlib.pyplot as plt

import os
import argparse
import warnings

# Convenient management
parser = argparse.ArgumentParser('Classifier of MNIST datasets!')
parser.add_argument('--dataset', '--d', type=str, default='mnist',
                    help="datset {'mnist', 'kmnist', 'emnist}. default: 'mnist'")
parser.add_argument('--classes', type=int, default=10,
                    help="Classification picture type. default: 10")
parser.add_argument('--buffer_size', type=int, default=5000,
                    help="Train dataset size. default: 5000.")
parser.add_argument('--batch_size', type=int, default=64,
                    help="one step train dataset size. default: 64")
parser.add_argument('--epochs', '--e', type=int, default=5,
                    help="Train epochs. default: 5")
parser.add_argument('--lr', '--learning_rate', type=float, default=0.001,
                    help='float >= 0. Learning rate. default: 0.001')
parser.add_argument('--b1', '--beta1', type=float, default=0.9,
                    help="float, 0 < beta < 1. Generally close to 1. default: 0.9")
parser.add_argument('--b2', '--beta2', type=float, default=0.999,
                    help="float, 0 < beta < 1. Generally close to 1. default: 0.999")
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help="float >= 0. Fuzz factor. defaults: 1e-8.")
parser.add_argument('--decay', type=float, default=0.,
                    help=" float >= 0. Learning rate decay over each update. defaults: 0. .")
parser.add_argument('--name', type=str,
                    help="Choose to use a neural network. option: {`mnist`, `alexnet`, `vgg16`, `vgg19`}")
parser.add_argument('--checkpoint_dir', '--dir', type=str,
                    help="Model save path.")

# Parses the parameters and prints them
args = parser.parse_args()
print(args)

# check model name
if args.name == 'lenet':
  model = LeNet(input_shape=(32, 32, 1),
                classes=args.classes)
elif args.name == 'alexnet':
  model = AlexNet(input_shape=(32, 32, 1),
                  classes=args.classes)
elif args.name == 'vgg16':
  model = VGG16(input_shape=(32, 32, 1),
                classes=args.classes)
elif args.name == 'vgg19':
  model = VGG19(input_shape=(32, 32, 1),
                classes=args.classes)
else:
  model = None

assert model is not None
if model is None:
  raise ValueError('If you use special model, please reference parameter help.'
                   '`python train.py --help / -h`.')

model.summary()

# check optimizer paras
if args.lr < 0:
  warnings.warn('float >= 0. Learning rate.')
if args.b1 <= 0:
  warnings.warn('float, 0 < beta < 1. Generally close to 1')
if args.b2 <= 0:
  warnings.warn('float, 0 < beta < 1. Generally close to 1.')
if args.epsilon < 0:
  warnings.warn('float >= 0. Fuzz factor.')
if args.decay < 0:
  warnings.warn('float >= 0. Learning rate decay over each update. ')

# define optimizer for Adam.
optimizer = tf.keras.optimizers.Adam(lr=args.lr,
                                     beta_1=args.b1,
                                     beta_2=args.b2,
                                     epsilon=args.epsilon,
                                     decay=args.decay)

# The cross entropy loss between the predicted value and the label was calculated
entropy = tf.losses.SparseCategoricalCrossentropy()

# setup model compile
model.compile(optimizer=optimizer,
              loss=entropy,
              metrics=['accuracy'])


def train():
  history = model.fit(train_dataset,
                      epochs=args.epochs,
                      validation_data=val_dataset)

  checkpoint_prefix = os.path.join(args.checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint.save(file_prefix=checkpoint_prefix)

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(acc, label='Training Accuracy')
  plt.plot(val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.ylabel('Accuracy')
  plt.ylim([min(plt.ylim()), 1])
  plt.title('Training and Validation Accuracy')
  plt.xlabel('epoch')

  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.ylabel('Cross Entropy')
  plt.ylim([0, 1.0])
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  plt.show()


if __name__ == '__main__':
  if args.dataset == 'mnist':
    assert args.classes == 10
    train_dataset, test_dataset, val_dataset = mnist.load_data_mnist()
  elif args.dataset == 'kmnist':
    assert args.classes == 10
    train_dataset, test_dataset, val_dataset = kmnist.load_data_kmnist()
  elif args.dataset == 'emnist':
    assert args.classes == 62
    train_dataset, test_dataset, val_dataset = emnist.load_data_emnist()
  else:
    exit(0)
    raise ValueError('If you use special dataset, please reference parameter help.'
                     '`python train.py --help / -h`.')

  train()
