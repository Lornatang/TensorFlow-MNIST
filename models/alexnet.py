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

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers


def alexnet(num_classes):
  model = keras.Sequential(name='AlexNet')
  model.add(layers.Conv2D(96,
                          [11, 11],
                          strides=4,
                          padding='same',
                          activation=tf.nn.relu,
                          input_shape=(224, 224, 3),
                          name='conv1'))
  model.add(layers.MaxPool2D((2, 2),
                             strides=(2, 2),
                             name='maxpool1'))

  model.add(layers.Conv2D(256,
                          (5, 5),
                          strides=1,
                          padding='same',
                          activation=tf.nn.relu,
                          name='conv2'))
  model.add(layers.MaxPool2D((3, 3),
                             strides=(2, 2),
                             name='maxpool2'))

  model.add(layers.Conv2D(384,
                          (3, 3),
                          strides=1,
                          padding='same',
                          activation=tf.nn.relu,
                          name='conv3'))
  model.add(layers.Conv2D(384,
                          (3, 3),
                          strides=1,
                          padding='same',
                          activation=tf.nn.relu,
                          name='conv4'))
  model.add(layers.Conv2D(256,
                          (3, 3),
                          strides=1,
                          padding='same',
                          activation=tf.nn.relu))
  model.add(layers.MaxPool2D((3, 3),
                             strides=(2, 2),
                             name='maxpool3'))

  model.add(layers.AvgPool2D((6, 6),
                             strides=(6, 6),
                             name='avgpool1'))

  model.add(layers.Flatten(name='flatten1'))

  model.add(layers.Dropout(0.3,
                           name='drop1'))
  model.add(layers.Dense(4096,
                         activation=tf.nn.relu))

  model.add(layers.Dropout(0.3,
                           name='drop2'))
  model.add(layers.Dense(4096,
                         activation=tf.nn.relu,
                         name='drop3'))

  model.add(layers.Dense(num_classes,
                         activation=tf.nn.softmax))

  return model
