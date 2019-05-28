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

"""Load MNIST dataset for use tfds lib"""

import tensorflow_datasets as tfds


def load_dataset(name='mnist', train_size=7, test_size=2, val_size=1):
  """ load every mnist dataset.

  Args:
    name:        "str",   dataset name.       default: mnist.
    train_size:  "int64", train dataset.      default:7
    test_size:   "int64", test dataset.       default:2
    val_size:    "int64", val dataset.        default:1

  Returns:
    dataset,

  """
  split_weights = (train_size, test_size, val_size)
  splits = tfds.Split.TRAIN.subsplit(weighted=split_weights)
  (train_dataset, test_dataset, val_dataset) = tfds.load(name,
                                                         split=list(splits),
                                                         as_supervised=True)

  return train_dataset, test_dataset, val_dataset
