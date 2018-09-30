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
import mynet
import data_helper
from data_helper import Preprocessor
from mynet import MyNet

MYNET = "vgg16"

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
labels_df.head(10)

labels_df_test = pd.read_csv(test_csv_file)
labels_df_test.head(10)

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
img_resize = (3, 128, 128)

# Split data into train/validation - percentage setting
validation_split_size = 0.2



# ## Data preprocessing
#----------------------------

preprocessor = Preprocessor(train_jpeg_dir, train_csv_file, test_jpeg_dir, test_csv_file,
                            img_resize[1::], validation_split_size)
preprocessor.init()

print("X_train/y_train length: {}/{}".format(len(preprocessor.X_train), len(preprocessor.y_train)))
print("X_val/y_val length: {}/{}".format(len(preprocessor.X_val), len(preprocessor.y_val)))
print("X_test/y_test length: {}/{}".format(len(preprocessor.X_test), len(preprocessor.y_test)))
preprocessor.y_map





# Constructing the model
#----------------------------

mynet = MyNet(net_selection=MYNET, img_dim=img_resize, num_classes=len(preprocessor.y_map))
mynet.model.summary()

# Load what's left off in last training
if os.path.isfile('weights/' + MYNET + '.weights.best.hdf5'):
    mynet.model.load_weights("weights/" + MYNET + ".weights.best.hdf5")
    print(MYNET, ": Weights loaded from last training...")
    

X_train, y_train = preprocessor.X_train, preprocessor.y_train
X_val, y_val = preprocessor.X_val, preprocessor.y_val

batch_size = 16
train_steps = len(X_train) / batch_size
val_steps = len(X_val) / batch_size

train_generator = preprocessor.get_train_generator(batch_size)
val_generator = preprocessor.get_val_generator(batch_size)

# Train the model
#----------------------------
from mymetrics import *
    
train_history = History()
callbacks = [train_history,
             EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=0, min_lr=1e-7, verbose=1),
             ModelCheckpoint(filepath='weights/' + MYNET + '.weights.epoch{epoch:02d}-valLoss{val_loss:.2f}.hdf5', verbose=1, save_best_only=False,
                             save_weights_only=True, mode='auto')]

# No serious training running on Jupyter notebooks, Run EPOCH
mynet.model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics = ['accuracy', fbeta, precision, recall])
train_history = mynet.model.fit_generator(train_generator, train_steps, epochs=2, verbose=1,
                    validation_data=val_generator, validation_steps=val_steps, callbacks=callbacks)


# ## Visualize Loss Curve

plt.figure(num=None, figsize=(8, 15), dpi=150, facecolor='w', edgecolor='k')
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['acc'])
plt.plot(train_history.history['fbeta'])
plt.plot(train_history.history['precision'])
plt.plot(train_history.history['recall'])
plt.plot(train_history.history['val_loss'])
plt.plot(train_history.history['val_acc'])
plt.plot(train_history.history['val_fbeta'])
plt.plot(train_history.history['val_precision'])
plt.plot(train_history.history['val_recall'])
plt.title('Training log')
plt.ylabel('loss/score')
plt.xlabel('epoch')
plt.legend(['train_loss', 'acc', 'fbeta', 'precision', 'recall', 'val_loss', 'val_acc', 'val_fbeta', 'val_precision', 'val_recall'], loc='upper left')
#plt.show()
plt.savefig(MYNET+".losses.png")

# Loading trained weights
mynet.model.load_weights("weights/" + MYNET + '.weights.epoch{epoch:02d}-valLoss{val_loss:.2f}.hdf5')
print("Weights loaded")
mynet.model.save("models/"+ MYNET + ".best.hdf5")


