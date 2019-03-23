import tensorflow as tf
from tensorflow._api.v1.keras.models import Sequential
from tensorflow._api.v1.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
# from tensorflow._api.v1.keras.layers.normalization import BatchNormalization
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

np.random.seed(1000)
#Instantiate an empty model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(200,200,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(200*200*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size

model.fit_generator(generator=train_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_gen,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
                    )
