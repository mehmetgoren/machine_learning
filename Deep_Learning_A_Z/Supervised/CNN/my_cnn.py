"""
    CNN
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

#step 1 Convolution
classifier.add(Conv2D(64,(3,3), input_shape=(128,128,3), activation="relu"))

#step 2 Pooling
classifier.add(MaxPool2D(pool_size=(2,2), data_format="channels_first"))


#lets add new conv layet to improve accuracy
classifier.add(Conv2D(64,(3,3), input_shape=(128,128,3), activation="relu"))
classifier.add(MaxPool2D(pool_size=(2,2), data_format="channels_first"))
classifier.add(Conv2D(64,(3,3), input_shape=(128,128,3), activation="relu"))
classifier.add(MaxPool2D(pool_size=(2,2), data_format="channels_first"))

#step 3 Flattining (a big single vector)
classifier.add(Flatten())

#step 4 Full Connection
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=64, activation="relu"))
classifier.add(Dense(units=32, activation="relu"))
classifier.add(Dense(units=16, activation="relu"))
classifier.add(Dense(units=8, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))


#Compiling the CNN
classifier.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])


#to avoid overfitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = 2000)




import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'