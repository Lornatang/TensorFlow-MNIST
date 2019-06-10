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

from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import utils


def AlexNet(include_top=True,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000):
  """ Instantiates the AlexNet architecture.

  Args:
    include_top: whether to include the 3 fully-connected
            layers at the top of the network.
    input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
    input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
    pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

  Returns:
    A Keras model instance.

  """
  if classes == 1000:
    raise ValueError('If use dataset is `imagenet`, please use it,'
                     'otherwise please use classifier images classes.')

  if input_shape == (32, 32, 1):
    raise ValueError('If use input shape is `32 * 32 * 1`, please don`t use it! '
                     'So you should change network architecture '
                     'or use input shape is `224 * 224 * 3`.')

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor
  x = layers.Conv2D(96, (11, 11),
                    strides=4,
                    activation=tf.nn.relu,
                    padding='same',
                    name='conv1')(img_input)
  x = layers.MaxPool2D((2, 2), strides=(2, 2), name='max_pool1')(x)

  x = layers.Conv2D(256, (5, 5),
                    strides=1,
                    activation=tf.nn.relu,
                    padding='same',
                    name='conv2')(x)
  x = layers.MaxPool2D((3, 3), strides=(2, 2), name='max_pool2')(x)

  x = layers.Conv2D(384, (3, 3),
                    strides=1,
                    activation=tf.nn.relu,
                    padding='same',
                    name='conv3')(x)
  x = layers.Conv2D(384, (3, 3),
                    strides=1,
                    activation=tf.nn.relu,
                    padding='same',
                    name='conv4')(x)
  x = layers.Conv2D(256, (3, 3),
                    strides=1,
                    activation=tf.nn.relu,
                    padding='same',
                    )(x)
  x = layers.MaxPool2D((3, 3), strides=(2, 2), name='max_pool3')(x)

  x = layers.AvgPool2D((6, 6), strides=(6, 6), name='avg_pool1')(x)

  if include_top:
    # Classification block
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(0.3, name='drop1')(x)
    x = layers.Dense(4096, activation=tf.nn.relu, name='fc1')(x)
    x = layers.Dropout(0.3, name='drop2')(x)
    x = layers.Dense(4096, activation=tf.nn.relu, name='fc2')(x)
    x = layers.Dense(classes, activation=tf.nn.softmax, name='predictions')(x)
  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D()(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input
  # Create model.

  model = models.Model(inputs, x, name='AlexNet')
  return model
