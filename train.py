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

from datasets.load_data import load_dataset
from models.make_model import *

import matplotlib.pyplot as plt
import numpy as np

import os

HEIGHT = 32
WIDTH = 32
checkpoint_dir = 'training_checkpoints'

EPOCHS = 5
BUFFER_SIZE = 5000
BATCH_SIZE = 64


def process_image(image, label, height=HEIGHT, width=WIDTH):
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


train_dataset, test_dataset, val_dataset = load_dataset(name='mnist')
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_dataset = train_dataset.map(process_image).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.map(process_image).batch(BATCH_SIZE)
val_dataset = val_dataset.map(process_image).batch(BATCH_SIZE)

model = lenet(num_classes=10)
model.summary()

model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


def train():
  history = model.fit(train_dataset,
                      epochs=EPOCHS,
                      validation_data=val_dataset)

  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
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

  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.ylabel('Cross Entropy')
  plt.ylim([0, 1.0])
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  plt.show()


train()
