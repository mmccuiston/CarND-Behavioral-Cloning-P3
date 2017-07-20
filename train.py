#!/usr/bin/env python

import csv
import cv2
import numpy as np
import time

def load_training_data(directory):
  lines = []
  with open("{}/driving_log.csv".format(directory)) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      lines.append(line)

  images = []
  measurements = []
  for line in lines:
    filename = line[0]
    center_image = cv2.imread(filename)
    images.append(center_image)
    measurement = float(line[3])
    measurements.append(measurement)
  return images, measurements


def preprocess(img):
  import tensorflow as tf
  height = 160
  width = 320
  print(height)
  print(width)
  gray = tf.image.rgb_to_grayscale(img)
  cropped = tf.image.crop_to_bounding_box(gray, int(height / 2), 0 ,int(height / 2), width)
  return cropped

def basic_model():
  from keras.models import Sequential
  from keras.layers import Flatten, Dense, Lambda
  from keras.layers.convolutional import Conv2D, Cropping2D, MaxPooling2D
  model = Sequential()
  model.add( Lambda(lambda img: img / 255.0 - 0.5, input_shape=(160,320,3) ) )
  model.add( Cropping2D( cropping=((50,20), (0,0))))
  model.add( Conv2D(6,5,5, activation="relu" ) )
  model.add( MaxPooling2D() )
  model.add( Conv2D(6,5,5, activation="relu" ) )
  model.add( MaxPooling2D() )
  model.add( Conv2D(6,5,5, activation="relu" ) )
  model.add( MaxPooling2D() )
  
  model.add( Flatten() )
  model.add( Dense(120) )
  model.add( Dense(84) )
  model.add( Dense(1) )
  return model

def advanced_model():
  from keras.models import Sequential
  from keras.layers import Flatten, Dense, Lambda
  from keras.layers.convolutional import Conv2D, Cropping2D
  model = Sequential()
  model.add( Lambda(lambda img: img / 255.0 - 0.5, input_shape=(160,320,3) ) )
  model.add( Cropping2D( cropping=((70,25), (0,0))))
  model.add( Conv2D(24,5,5, subsample=(2,2), activation="relu" ) )
  model.add( Conv2D(36,5,5, subsample=(2,2), activation="relu" ) )
  model.add( Conv2D(48,5,5, subsample=(2,2), activation="relu" ) )
  model.add( Conv2D(64,3,3, activation="relu" ) )
  model.add( Conv2D(64,3,3, activation="relu" ) )
  model.add( Flatten() )
  model.add( Dense(100) )
  model.add( Dense(50) )
  model.add( Dense(10) )
  model.add( Dense(1) )
  return model

def train_model(images, measurements):

  X_train = np.array(images)
  y_train = np.array(measurements)



  #model = basic_model()
  model = advanced_model()
  model.compile(loss='mse', optimizer='adam')
  model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

  return model


def augment(images, measurements):
  flipped_images = []
  flipped_measurements = []
  for i,image in enumerate(images):
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = measurements[i] * -1.0 

  return images + flipped_images, measurements + flipped_measurements

images, measurements = load_training_data("./training-data")
images, measurements = augment(images, measurements)
#images = [preprocess(img) for img in images]

import matplotlib.pyplot as plt
plt.figure(figsize=(24,12))
plt.imshow(images[0])
#plt.show()

model = train_model(images, measurements)
model.save('model.h5')
