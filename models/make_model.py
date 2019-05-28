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

"""There are various model implementations for CNN"""

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers


def lenet(num_classes):
  model = keras.Sequential(name='LeNet-5')
  model.add(layers.Conv2D(6,
                          (5, 5),
                          padding='same',
                          activation=tf.nn.relu,
                          input_shape=(32, 32, 1),
                          name='conv1'))
  model.add(layers.MaxPooling2D((2, 2),
                                strides=(2, 2),
                                name='maxpool1'))
  model.add(layers.Conv2D(16,
                          (3, 3),
                          padding='same',
                          activation=tf.nn.relu,
                          name='conv2'))
  model.add(layers.MaxPooling2D((2, 2),
                                strides=(2, 2),
                                name='maxpool2'))

  model.add(layers.Flatten(name='flatten1'))

  model.add(layers.Dense(120,
                         activation=tf.nn.relu,
                         name='dense1'))
  model.add(layers.Dense(84,
                         activation=tf.nn.relu,
                         name='dense2'))
  model.add(layers.Dense(num_classes,
                         activation=tf.nn.softmax,
                         name='dense3'))

  return model


def alexnet(num_classes):
  models = keras.Sequential(name='AlexNet')
  models.add(layers.Conv2D(96,
                           (11, 11),
                           strides=4,
                           activation=tf.nn.relu,
                           padding='same',
                           input_shape=(224, 224, 3),
                           name='conv1'))
  models.add(layers.MaxPool2D((2, 2),
                              strides=(2, 2),
                              name='maxpool1'))

  models.add(layers.Conv2D(256,
                           (5, 5),
                           strides=1,
                           activation=tf.nn.relu,
                           padding='same',
                           name='conv2'))
  models.add(layers.MaxPool2D((3, 3),
                              strides=(2, 2),
                              name='maxpool2'))

  models.add(layers.Conv2D(384,
                           (3, 3),
                           strides=1,
                           activation=tf.nn.relu,
                           padding='same',
                           name='conv3'))
  models.add(layers.Conv2D(384,
                           (3, 3),
                           strides=1,
                           activation=tf.nn.relu,
                           padding='same',
                           name='conv4'))
  models.add(layers.Conv2D(256,
                           (3, 3),
                           strides=1,
                           activation=tf.nn.relu,
                           padding='same'))
  models.add(layers.MaxPool2D((3, 3),
                              strides=(2, 2),
                              name='maxpool3'))

  models.add(layers.AvgPool2D((6, 6),
                              strides=(6, 6),
                              name='avgpool1'))

  models.add(layers.Flatten(name='flatten1'))

  models.add(layers.Dropout(0.3,
                            name='drop1'))
  models.add(layers.Dense(4096,
                          activation=tf.nn.relu))

  models.add(layers.Dropout(0.3,
                            name='drop2'))
  models.add(layers.Dense(4096,
                          activation=tf.nn.relu,
                          name='drop3'))

  models.add(layers.Dense(num_classes,
                          activation=tf.nn.softmax))

  return models
