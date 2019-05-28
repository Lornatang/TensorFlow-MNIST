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

"""VGG16 model for TensorFlow."""

import tensorflow as tf

from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import utils


def VGG19(include_top=True,
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000):
  """ Instantiates the LeNet-5 architecture.

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
    raise ValueError('If use dataset is `imagenet`, please use it'
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

  # Block 1
  x = layers.Conv2D(64, (3, 3),
                    activation=tf.nn.relu,
                    padding='same',
                    name='block1_conv1')(img_input)
  x = layers.Conv2D(64, (3, 3),
                    activation=tf.nn.relu,
                    padding='same',
                    name='block1_conv2')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

  # Block 2
  x = layers.Conv2D(128, (3, 3),
                    activation=tf.nn.relu,
                    padding='same',
                    name='block2_conv1')(x)
  x = layers.Conv2D(128, (3, 3),
                    activation=tf.nn.relu,
                    padding='same',
                    name='block2_conv2')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

  # Block 3
  x = layers.Conv2D(256, (3, 3),
                    activation=tf.nn.relu,
                    padding='same',
                    name='block3_conv1')(x)
  x = layers.Conv2D(256, (3, 3),
                    activation=tf.nn.relu,
                    padding='same',
                    name='block3_conv2')(x)
  x = layers.Conv2D(256, (3, 3),
                    activation=tf.nn.relu,
                    padding='same',
                    name='block3_conv3')(x)
  x = layers.Conv2D(256, (3, 3),
                    activation=tf.nn.relu,
                    padding='same',
                    name='block3_conv4')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

  # Block 4
  x = layers.Conv2D(512, (3, 3),
                    activation=tf.nn.relu,
                    padding='same',
                    name='block4_conv1')(x)
  x = layers.Conv2D(512, (3, 3),
                    activation=tf.nn.relu,
                    padding='same',
                    name='block4_conv2')(x)
  x = layers.Conv2D(512, (3, 3),
                    activation=tf.nn.relu,
                    padding='same',
                    name='block4_conv3')(x)
  x = layers.Conv2D(512, (3, 3),
                    activation=tf.nn.relu,
                    padding='same',
                    name='block4_conv4')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

  # Block 5
  x = layers.Conv2D(512, (3, 3),
                    activation=tf.nn.relu,
                    padding='same',
                    name='block5_conv1')(x)
  x = layers.Conv2D(512, (3, 3),
                    activation=tf.nn.relu,
                    padding='same',
                    name='block5_conv2')(x)
  x = layers.Conv2D(512, (3, 3),
                    activation=tf.nn.relu,
                    padding='same',
                    name='block5_conv3')(x)
  x = layers.Conv2D(512, (3, 3),
                    activation=tf.nn.relu,
                    padding='same',
                    name='block5_conv4')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

  if include_top:
    # Classification block
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(4096, activation=tf.nn.relu, name='fc1')(x)
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
  model = models.Model(inputs, x, name='vgg16')

  return model
