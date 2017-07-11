#!/usr/bin/env python

import csv
import cv2
import numpy as np

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




def train_model(images, measurements):

  X_train = np.array(images)
  y_train = np.array(measurements)


  from keras.models import Sequential
  from keras.layers import Flatten, Dense, Activation, Lambda


  model = Sequential()
  model.add(Flatten(input_shape=(160,320,3)))
  model.add(Lambda(lambda x: x / 256.0 - 0.5))
  model.add(Dense(50))
  model.add(Activation('relu'))
  model.add(Dense(1))
  model.add(Activation('relu'))

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
model = train_model(images, measurements)
model.save('model.h5')
