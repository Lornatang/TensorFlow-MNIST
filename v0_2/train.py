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
from v0_2.dataset import load_data
from v0_2.models import *

import tensorflow as tf

# plt pic
import matplotlib.pyplot as plt

import os
import argparse
import warnings

# Convenient management
parser = argparse.ArgumentParser('Classifier of Cats_VS_Dogs datasets!')
parser.add_argument('--dataset', '--d', type=str, default='caltech101',
                    help="datset {'mnist', 'kmnist', 'emnist}. default: 'mnist'")

parser.add_argument('--height', '--h', type=int, default=224,
                    help='Image height. default: 224')
parser.add_argument('--width', '--w', type=int, default=224,
                    help='Image width.  default: 224')
parser.add_argument('--channels', '--c', type=int, default=3,
                    help='Image color RBG. default: 3')

parser.add_argument('--classes', type=int, default=102,
                    help="Classification picture type. default: 102")
parser.add_argument('--buffer_size', type=int, default=1000,
                    help="Train dataset size. default: 1000.")
parser.add_argument('--batch_size', type=int, default=16,
                    help="one step train dataset size. default: 16")
parser.add_argument('--epochs', '--e', type=int, default=10,
                    help="Train epochs. default: 10")

parser.add_argument('--lr', '--learning_rate', type=float, default=0.0001,
                    help='float >= 0. Learning rate. default: 0.0001')
parser.add_argument('--b1', '--beta1', type=float, default=0.9,
                    help="float, 0 < beta < 1. Generally close to 1. default: 0.9")
parser.add_argument('--b2', '--beta2', type=float, default=0.999,
                    help="float, 0 < beta < 1. Generally close to 1. default: 0.999")
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help="float >= 0. Fuzz factor. defaults: 1e-8.")
parser.add_argument('--decay', type=float, default=0.,
                    help=" float >= 0. Learning rate decay over each update. defaults: 0. .")

parser.add_argument('--name', type=str, default='alexnet',
                    help="Choose to use a neural network. option: {`mnist`, `alexnet`, `vgg16`, `vgg19`}")
parser.add_argument('--checkpoint_dir', '--dir', type=str, default='training_checkpoint',
                    help="Model save path.")

parser.add_argument('--dis', type=bool, default=False,
                    help='display matplotlib? default: False.')

# Parses the parameters and prints them
args = parser.parse_args()
print(args)


# check model name
if args.name == 'lenet':
  model = LeNet(input_shape=(32, 32, 1),
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
optimizer = tf.optimizers.Adam(lr=args.lr,
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
  assert args.classes == 102
  train_dataset, val_dataset, test_dataset = load_data()
  train()
