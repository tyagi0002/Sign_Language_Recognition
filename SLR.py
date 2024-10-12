from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys
import tensorflow as tf
import pandas as pd 
import numpy as np
import tensorflow_hub as hub
import os

from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from keras import optimizers

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)

print("Version ", tf.__version__)
print("Eager mode:", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is","available" if tf.test.is_gpu_available() else"Not Available")

base_dir = "/kaggle/input/american-sign-language-recognition"
train_path = "/kaggle/input/american-sign-language-recognition/training_set"
val_path = "/kaggle/input/american-sign-language-recognition/test_set"

import time
from os.path import exists

def count(dir, counter=0):
    "returns number of files in dir and subdirs"
    for pack in os.walk(dir):
        for f in pack[2]:
            counter += 1
    return dir + " : " + str(counter) + " files"

print('total images for training :', count(train_path))
print('total images for validation :', count(val_path))

%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nrows = 4
ncols = 4

# for iternating over images
pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols *4, nrows*4)

train_A = os.listdir(os.path.join(train_path,'A'))

pic_index += 8
next_pix = [os.path.join(train_path,'A', fname)
                for fname in train_A[pic_index-8:pic_index]]
for i, img_path in enumerate(next_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows,ncols,i +1)
  #sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

IMAGE_SHAPE = (244, 244)
BATCH_SIZE = 64

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    val_path, 
    shuffle=False, 
    seed=42,
    color_mode="rgb", 
    class_mode="categorical",
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE)

do_data_augmentation = True
if do_data_augmentation:
  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rescale = 1./255,
      rotation_range=40,
      horizontal_flip=True,
      width_shift_range=0.2, 
      height_shift_range=0.2,
      shear_range=0.2, 
      zoom_range=0.2,
      fill_mode='nearest' )
else:
  train_datagen = validation_datagen
  
train_generator = train_datagen.flow_from_directory(
    train_path,  
    shuffle=True, 
    seed=42,
    color_mode="rgb", 
    class_mode="categorical",
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE)

classes=train_generator.class_indices
print(classes.keys())

class MyCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,log = {}):
    if(log.get('accuracy')> 0.99):
      if(log.get('val_accuracy')>0.99):
        print("\n Reached 99% Accuracy for both train and val.")
        self.model.stop_training = True

callbacks = MyCallback()

!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf.dim_ordering_tf_kernels.notop.h5

from  tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
local_weights_file = '/tmp/inception_v3_weights_tf.dim_ordering_tf_kernels.notop.h5'

pre_trained_model = InceptionV3(
                                input_shape = (244,244,3),
                                include_top= False,
                                weights = None
)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print(f'The shape of the last layer is {last_layer.output_shape}')
output_layer = last_layer.output

import tensorflow as tf
from tensorflow.keras import layers

x = tf.keras.layers.Flatten()(output_layer)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
#x = tf.keras.layers.Dropout(0.2)(x)ener
x = tf.keras.layers.Dense(40, activation='softmax')(x)

model = Model(pre_trained_model.input, x,name="Signs_Inception_model")

LEARNING_RATE = 0.001 #@param {type:"number"}

model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = LEARNING_RATE),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(train_generator , verbose=2 ,
                    epochs=5 , validation_data=validation_generator,
                    steps_per_epoch=train_generator.samples//train_generator.batch_size,
                    validation_steps=validation_generator.samples//validation_generator.batch_size,
                    callbacks = [callbacks],
                    use_multiprocessing= True)
