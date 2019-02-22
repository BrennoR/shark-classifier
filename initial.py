import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

data = []

for subdir, dirs, files in os.walk('images'):
    for file in files:
        # print(os.path.join(subdir, file)[7:])
        # data.append([os.path.join(subdir, file)[7:], file[:2]])
        data.append([file, file[:2]])

pd_data = pd.DataFrame(data, columns=['filename', 'class'])
pd_data['class'] = pd.Categorical(pd_data['class'])

datagen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=True,
                                                          featurewise_std_normalization=True,
                                                          rotation_range=20,
                                                          width_shift_range=0.2,
                                                          height_shift_range=0.2,
                                                          horizontal_flip=True,
                                                          rescale=1./255.,
                                                          validation_split=0.2)

train_gen = datagen.flow_from_dataframe(pd_data, directory='./images/', subset='training', target_size=(200, 200))
valid_gen = datagen.flow_from_dataframe(pd_data, directory='./images/', subset='validation', target_size=(200, 200))

from tensorflow._api.v1.keras.models import Sequential
from tensorflow._api.v1.keras.layers import Dense, Conv2D, Flatten

model = Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size

model.fit_generator(generator=train_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_gen,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
)

pred = model.predict_generator(train_gen,
steps=STEP_SIZE_VALID,
verbose=1)

# print(pred)
