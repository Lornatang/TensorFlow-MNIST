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
