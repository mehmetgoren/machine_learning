""""
    CNN
"""

from keras import backend as K
gpus = K.tensorflow_backend._get_available_gpus()

print(len(gpus))

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())




from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

#initialising CNN
classifier = Sequential()

#step 1 - convolution
#classifier.add(Convolution2D(64,3,3, input_shape=(256,256,3), activation="relu"))#64=feature detectors, 3=3X3 deimension matrix.input_shape=3 RGB.
classifier.add(Conv2D(64,(3,3), input_shape=(256,256,3), activation="relu"))

#step 2  - Max Pooling. reducing nhumber of layers
classifier.add(MaxPool2D(pool_size=(2,2)))


#added new cnn to improve success
classifier.add(Conv2D(64,(3,3), input_shape=(256,256,3), activation="relu"))
classifier.add(MaxPool2D(pool_size=(2,2)))


#step 3  - Flatting
classifier.add(Flatten())

#step 4 - Full Connection
classifier.add(Dense(units=128, activation="relu"))
#classifier.add(Dense(units=64, activation="relu"))
#classifier.add(Dense(units=256, activation="relu"))
#classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))


#Compiling the CNN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])#loss da birden fazla uutput olsaydı categorical derdik.


#lets get images
from keras.preprocessing.image import ImageDataGenerator
import scipy.ndimage

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

trainig_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(256, 256),
        batch_size=64,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(256, 256),
        batch_size=64,
        class_mode='binary')

classifier.fit_generator(
        trainig_set,
        steps_per_epoch=8000,#resim sayısı
        epochs=1,
        validation_data=test_set,
        validation_steps=2000)#test resim sayısıü


#bu model %75 .çıkıyor. Arttırmak için başarıyı ann veya cnn ekle.