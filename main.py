#Importing Necessay Libraries and APIS
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers

#Creating paths for data preprocessing
data_train_path = "Data_Train"
data_validate_path = "Data_Validate"
data_test_path = "Data_Test"

#Setting image size for preprocessing
img_width = 331
img_height = 331

#Function for training data
def train_data(data_train_path, img_width, img_height):
    global data_train
    data_train = tf.keras.utils.image_dataset_from_directory(
        data_train_path,
        shuffle=True,
        image_size=(img_width,img_height),
        batch_size=64,
        validation_split=False
        )

    global data_categories_train
    data_categories_train = data_train.class_names

    print("~~~~~~~~~~~~~~~~~~DATA TRAIN~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~DATA TRAIN~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~DATA TRAIN~~~~~~~~~~~~~~~~~~")

    return data_categories_train

#Function for validating data
def validate_data(validate_train_path, img_width, img_height):
    global data_validate
    data_validate = tf.keras.utils.image_dataset_from_directory(
        validate_train_path,
        shuffle=False,
        image_size=(img_width,img_height),
        batch_size=32,
        validation_split=False
        )
    
    global data_categories_validate
    data_categories_validate = data_validate.class_names

    print("~~~~~~~~~~~~~~~~~~DATA VALIDATE~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~DATA VALIDATE~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~DATA VALIDATE~~~~~~~~~~~~~~~~~~")

    return data_categories_validate

#Function for testing data
def test_data(test_train_path, img_width, img_height):
    global data_test
    data_test = tf.keras.utils.image_dataset_from_directory(
        test_train_path,
        shuffle=False,
        image_size=(img_width,img_height),
        batch_size=32,
        validation_split=False
        )

    global data_categories_test
    data_categories_test = data_test.class_names

    print("~~~~~~~~~~~~~~~~~~DATA TEST~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~DATA TEST~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~DATA TEST~~~~~~~~~~~~~~~~~~")

    return data_categories_test

def plot_images(train_data, data_categories_train):
    print("Starting process")
    plt.figure(figsize=(10,10))
    for image, labels in train_data.take(1):
        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.imshow(image[i].numpy().astype('uint8'))
            plt.title(data_categories_train[labels[i]])
            plt.show()
    return "Completed Printing of Images"

#This is what will be used to actually output what I need
print("starting preprocessing")
train_data(data_train_path, img_width, img_height)
validate_data(data_validate_path, img_width, img_height)
test_data(data_test_path, img_width, img_height)
print("completed preprocessing")

#Printing Images from training data using matplotlib
print("Printing Images using matplotlib")
plot_images(data_train, data_categories_train)
