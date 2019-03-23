import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import ceil
from data_manipulation import X_train, X_test, y_train, y_test

train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
test_data.reset_index(inplace=True)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.,
                                                          horizontal_flip=True,
                                                          vertical_flip=True,
                                                          # featurewise_center=True,
                                                          # featurewise_std_normalization=True,
                                                          # # zca_whitening=True,
                                                          # width_shift_range=0.2,
                                                          # height_shift_range=0.2,
                                                          # rotation_range=90,
                                                          validation_split=0.18)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)


train_gen = datagen.flow_from_dataframe(train_data, directory='./images', subset='training', target_size=(200, 200))
valid_gen = datagen.flow_from_dataframe(train_data, directory='./images', subset='validation', target_size=(200, 200))
test_gen = test_datagen.flow_from_dataframe(test_data, directory='./images', target_size=(200, 200))

from tensorflow._api.v1.keras.models import Sequential
from tensorflow._api.v1.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

model = Sequential()

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size
STEP_SIZE_TEST = ceil(test_gen.n / test_gen.batch_size)

history = model.fit_generator(generator=train_gen,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_gen,
                              validation_steps=STEP_SIZE_VALID,
                              epochs=40
                              )

model.evaluate_generator(generator=valid_gen)

test_gen.reset()
pred = model.predict_generator(test_gen,
                               steps=STEP_SIZE_TEST,
                               verbose=1)


predicted_class_indices = np.argmax(pred, axis=1)

labels = (train_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames = test_gen.filenames
results = pd.DataFrame({"Filename": filenames,
                        "Predictions": predictions})

final = []
for f, y in zip(results['Filename'], results['Predictions']):
    if 'gw' in f:
        final.append([0, y])
    else:
        final.append([1, y])

from sklearn.metrics import accuracy_score

print(accuracy_score(final[0], final[1]))

results.to_csv("results.csv", index=False)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'])
plt.show()
