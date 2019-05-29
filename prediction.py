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
import matplotlib.pyplot as plt
from PIL import Image

from models import LeNet

import argparse

parser = argparse.ArgumentParser('Prediction mnist label')

parser.add_argument('--path', type=str,
                    help='Image path, best input abs path. `./datasets/5.png`')
parser.add_argument('--num_classes', type=int, default=10,
                    help="Classification picture type. default: 10")
parser.add_argument('--checkpoint_dir', '--dir', type=str,
                    help="Model save path.")
args = parser.parse_args()


def process_image(image):
  """ process image ops.

    Args:
      image: 'input tensor'.

    Returns:
      tensor

    """
  # read img to string.
  image = tf.io.read_file(image)
  # decode png to tensor
  image = tf.image.decode_image(image, channels=1)
  # convert image to float32
  image = tf.cast(image, tf.float32)
  # image norm.
  image = image / 255.
  # image resize model input size.
  image = tf.image.resize(image, (32, 32))
  return image


def prediction(image):
  """ prediction image label.

  Args:
    image: 'input tensor'.

  Returns:
    'int64', label

  """
  image = process_image(image)
  # Add the image to a batch where it's the only member.
  image = (tf.expand_dims(image, 0))

  model = LeNet(input_shape=(32, 32, 1),
                classes=args.num_classes)

  print(f"==========================================")
  print(f"Loading model.............................")
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir))
  print(f"Load model successful!")
  print(f"==========================================")
  print(f"Start making predictions about the picture.")
  print(f"==========================================")
  predictions = model(image)
  classes = tf.argmax(predictions[0])
  print(f"label is : {classes}")

  image = Image.open(args.path)
  plt.figure(figsize=(4, 4))
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(image, cmap='gray')
  plt.xlabel(int(classes))
  plt.show()


if __name__ == '__main__':
  prediction(args.path)
