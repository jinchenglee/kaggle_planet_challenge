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





# Constructing the model
#----------------------------

mynet = MyNet(net_selection=MYNET, img_dim=img_resize)
mynet.model.summary()
mynet.model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics = ['accuracy'])


X_train, y_train = preprocessor.X_train, preprocessor.y_train
X_val, y_val = preprocessor.X_val, preprocessor.y_val

batch_size = 32
train_steps = len(X_train) / batch_size
val_steps = len(X_val) / batch_size

train_generator = preprocessor.get_train_generator(batch_size)
val_generator = preprocessor.get_val_generator(batch_size)

# Train the model
#----------------------------

train_history = History()
callbacks = [train_history,
             EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=0, min_lr=1e-7, verbose=1),
             ModelCheckpoint(filepath='weights/weights.best.hdf5', verbose=1, save_best_only=True,
                             save_weights_only=True, mode='auto')]

# No serious training running on Jupyter notebooks, Run EPOCH
train_history = mynet.model.fit_generator(train_generator, train_steps, epochs=25, verbose=1,
                    validation_data=val_generator, validation_steps=val_steps, callbacks=callbacks)


# ## Visualize Loss Curve

plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
plt.savefig(MYNET+".losses.png")


# Evaluate
#----------------------------

# Loading trained weights
mynet.model.load_weights("weights/weights.best.hdf5")
print("Weights loaded")


# Make predictions to get raw prediction values of each class/sample
predictions, x_test = mynet.predict(preprocessor, batch_size=32)
print("Predictions shape: {}\nFiles name shape: {}\n1st predictions ({}) entry:\n{}".format(predictions.shape, 
                                    x_test.shape, x_test[0], predictions[0]))

# Setting threshold for each class
thresholds = [0.2] * len(labels_set)
# Map raw predictions to label names
predicted_labels = mynet.map_predictions(preprocessor, predictions, thresholds)


# ### Peep into predictions
import re

# Look at predicted_labels vs. GT
###################################
# NO REAL GROUND TRUTH exists!!!
# Kaggle doesn't provide GT for test data.
# Used some fake ones instead. 
###################################
labels_dict_test = dict()
for fn,tags in labels_df_test.values:
    labels_dict_test[fn] = tags
    
plt.rc('axes', grid=False)
_, axs = plt.subplots(5, 2, sharex='col', sharey='row', figsize=(15, 20))
axs = axs.ravel()

for j in range(10):
    i = random.randint(0, len(predicted_labels))
    img = mpimg.imread(x_test[i])
    
    x_test_stripped = ''
    matchObj = re.match( r'.*(test_\d*).jpg.*', x_test[i])
    x_test_stripped = matchObj.group(1)

    labels=labels_dict_test[x_test_stripped]
    print(j, x_test[i], predicted_labels[i], labels)
    axs[j].imshow(img)
    axs[j].set_title('Pred:{}'.format(predicted_labels[i]))
    axs[j].set_xlabel('GT:{}'.format(labels))
#plt.show()
plt.savefig(MYNET+".peep_test_data.png")


# Evaluate loss and metrics
import psutil

batch_size=32
print("model metrics_name:", mynet.model.metrics_names)
my_loss, my_metric = mynet.model.evaluate_generator(preprocessor.get_prediction_generator(batch_size),
                         len(preprocessor.X_test) / batch_size, workers = (psutil.cpu_count()-1))
print("my_loss=", my_loss, "my_metric=", my_metric)


fbeta_score = mynet.fbeta(preprocessor, mode=1)
print("fbeta_score (training data) = ", fbeta_score)


# TODO: No thresholding applied yet! Use 0.2 as default.
fbeta_score = mynet.fbeta(preprocessor, mode=2)
print("fbeta_score (validation data) = ", fbeta_score)

##----------------------------------------
## TEST data score is expected to be low, as there NO REAL GROUND TRUTH!!!
##----------------------------------------
fbeta_score = mynet.fbeta(preprocessor)
print("fbeta_score (test data) = ", fbeta_score)


tags_list = [None] * len(predicted_labels)
for i, tags in enumerate(predicted_labels):
    tags_list[i] = ' '.join(map(str, tags))

final_data = [[filename.split(".")[0], tags] for filename, tags in zip(x_test, tags_list)]

final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
print("Predictions rows:", final_df.size)
final_df.head()

final_df.to_csv('../submission_file.csv', index=False)


#Manually calculate fbeta scores, to confirm above calcuations are correct.
from sklearn.metrics import fbeta_score

preprocessor.y_map

test_img_y_prediction = np.zeros_like(preprocessor.y_test)

for i in range(len(preprocessor.X_test)):
    # Load image
    test_img_name = preprocessor.X_test[i]
    test_img_y = preprocessor.y_test[i]
    test_img_x, test_img_y = preprocessor._val_transform_to_matrices((test_img_name, test_img_y))
    # Add dimension 'batch'
    test_img_x = test_img_x.reshape(-1, 128, 128, 3)
    
    # Make prediction
    test_img_y_prediction[i] = mynet.model.predict(test_img_x)[0]
    
    # Calculate fbeta score
    #score = fbeta_score(test_img_y, test_img_y_prediction[0] > 0.2, beta=2)

    #if score < 0.8:
    #    print("filename=", test_img_name, "score=", score)

print("fbeta_avg_samples=", fbeta_score(np.array(preprocessor.y_test), test_img_y_prediction > 0.2, beta=1, average='samples'))
print("fbeta_avg_micro=", fbeta_score(np.array(preprocessor.y_test), test_img_y_prediction > 0.2, beta=1, average='micro'))
print("fbeta_avg_macro=", fbeta_score(np.array(preprocessor.y_test), test_img_y_prediction > 0.2, beta=1, average='macro'))

