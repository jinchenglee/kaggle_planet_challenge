#!/usr/bin/env python
# coding: utf-8

# ## Environment preparation
import sys
sys.path.append('../src')

import os
import gc
import bcolz
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History
from itertools import chain
import random
import vgg16
import resnet50
import data_helper
from data_helper import Preprocessor

mynet = vgg16

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Let pandas to print full name of data contents, instead of 'blah...'
pd.set_option('display.max_colwidth', -1)

tf.__version__

# Data input
destination_path = "../input/"
is_datasets_present = True

datasets_path = data_helper.get_jpeg_data_files_paths()
for dir_path in datasets_path:
    if os.path.exists(dir_path):
        is_datasets_present = True

if not is_datasets_present:
    print("Not all datasets are present.")
else:
    print("All datasets are present.")


# ## Loading data 
#----------------------------

# Training and test data loading
train_jpeg_dir, test_jpeg_dir, train_csv_file, test_csv_file = data_helper.get_jpeg_data_files_paths()
labels_df = pd.read_csv(train_csv_file)
labels_df.head()

labels_df_test = pd.read_csv(test_csv_file)
labels_df_test.head()

# Brief look at data
labels_list = list(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values]))
labels_set = set(labels_list)
print("There are {} unique labels including {}".format(len(labels_set), labels_set))

images_title = [labels_df[labels_df['tags'].str.contains(label)].iloc[i]['image_name'] + '.jpg' 
                for i, label in enumerate(labels_set)]

labels_dict = dict()
for fn,tags in labels_df.values:
    labels_dict[fn] = tags

# Resize input
img_resize = (256, 256, 3)

# Split data into train/validation - percentage setting
validation_split_size = 0.2



# ## Data preprocessing
#----------------------------

preprocessor = Preprocessor(train_jpeg_dir, train_csv_file, test_jpeg_dir, test_csv_file, 
                                  img_resize[:2], validation_split_size)
preprocessor.init()

print("X_train/y_train length: {}/{}".format(len(preprocessor.X_train), len(preprocessor.y_train)))
print("X_val/y_val length: {}/{}".format(len(preprocessor.X_val), len(preprocessor.y_val)))
print("X_test/y_test length: {}/{}".format(len(preprocessor.X_test), len(preprocessor.y_test)))
preprocessor.y_map


history = History()
callbacks = [history,
             EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=0, min_lr=1e-7, verbose=1),
             ModelCheckpoint(filepath='weights/weights.best.hdf5', verbose=1, save_best_only=True,
                             save_weights_only=True, mode='auto')]

X_train, y_train = preprocessor.X_train, preprocessor.y_train
X_val, y_val = preprocessor.X_val, preprocessor.y_val




# ## Constructing the model
#----------------------------

model = mynet.create_model(img_dim=img_resize)
model.summary()

batch_size = 32
train_steps = len(X_train) / batch_size
val_steps = len(X_val) / batch_size

train_generator = preprocessor.get_train_generator(batch_size)
val_generator = preprocessor.get_val_generator(batch_size)

model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics = ['accuracy'])
# model.compile(optimizer=Adam(lr=1e-4), loss='mean_absolute_error', metrics = ['accuracy'])

history = model.fit_generator(train_generator, train_steps, epochs=25, verbose=1,
                    validation_data=val_generator, validation_steps=val_steps, callbacks=callbacks)


# ## Visualize Loss Curve

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
plt.savefig("vgg16.losses.png")
