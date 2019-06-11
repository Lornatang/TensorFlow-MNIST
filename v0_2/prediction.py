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

from v0_2.models import *
from v0_2.dataset import get_label_name

import tensorflow as tf

import argparse
import time

parser = argparse.ArgumentParser('Prediction mnist label')

parser.add_argument('--height', type=int, default=32,
                    help='Image height. default: 224')
parser.add_argument('--width', type=int, default=32,
                    help='Image width.  default: 224')
parser.add_argument('--channels', type=int, default=1,
                    help='Image color RBG. default: 1')
parser.add_argument('--classes', type=int, default=10,
                    help="Classification picture type. default: 10")
parser.add_argument('--checkpoint_dir', '--dir', type=str,
                    help="Model save path.")
args = parser.parse_args()

label_names = get_label_name()
label_names = label_names.features['label'].int2str

model = LeNet(input_shape=(32, 32, 1),
              classes=args.classes)


print(f"==========================================")
print(f"Loading model.............................")
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir))
print(f"Load model successful!")
print(f"==========================================")
print()


def process_image(image, height=args.height, width=args.width):
  """ process image ops.

    Args:
      image: 'input tensor'.
      height: 'int64' img height.
      width: 'int64' img width.

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
  image = tf.image.resize(image, (height, width))

  return image


def prediction(img):
  """ prediction image label.

  Args:
    img: 'input tensor'.

  Returns:
    'int64', label

  """
  image = process_image(img)
  # Add the image to a batch where it's the only member.
  image = (tf.expand_dims(image, 0))

  print(f"Start making predictions about the picture......")
  start = time.time()

  predictions = model(image)
  classes = int(tf.argmax(predictions[0]))
  print("done.")
  print()
  print(f"Label is : {label_names(classes)} times: {time.time() - start:.4} sec")
  print()


if __name__ == '__main__':
  while True:
    a = input("files(input '0' or 'any str' to exit.):")
    if a == '0':
      print("Successful exit. return status 0.")
      exit(0)
    else:
      a = a[:-1]
      a = tf.cast(a, tf.string)
      prediction(a)
