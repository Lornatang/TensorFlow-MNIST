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
import tensorflow_datasets as tfds


def process_image_with_kmnist(image, label, height=32, width=32):
  """ Resize the images to a fixes input size,
      and rescale the input channels to a range of [-1,1].

  Args:
    image: "tensor, float32", image input.
    label: "tensor, int64",   image label.
    height: "int64", (224, 224, 3) -> (height, 224, 3).
    width: "int64",  (224, 224, 3) -> (224, width, 3).

  Returns:
    image input, image label.

  """
  image = tf.cast(image, tf.float32)
  image = image / 255.
  image = tf.image.resize(image, (height, width))
  return image, label


def load_data_kmnist(name='kmnist', train_size=7, test_size=2, val_size=1, buffer_size=1000, batch_size=32):
  """ load every mnist dataset.

  Args:
    name:        "str",   dataset name.       default: 'kmnist'.
    train_size:  "int64", train dataset.      default:7
    test_size:   "int64", test dataset.       default:2
    val_size:    "int64", val dataset.        default:1
    buffer_size: "int64", dataset size.       default:1000.
    batch_size:  "int64", batch size.         default:32

  Returns:
    dataset,

  """
  split_weights = (train_size, test_size, val_size)
  splits = tfds.Split.TRAIN.subsplit(weighted=split_weights)
  (train_dataset, test_dataset, val_dataset) = tfds.load(name,
                                                         split=list(splits),
                                                         as_supervised=True)

  train_dataset = train_dataset.map(process_image_with_kmnist).shuffle(buffer_size).batch(batch_size)
  test_dataset = test_dataset.map(process_image_with_kmnist).batch(batch_size)
  val_dataset = val_dataset.map(process_image_with_kmnist).batch(batch_size)

  return train_dataset, test_dataset, val_dataset
