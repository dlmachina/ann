#convolutional neural network 
#building a cnn
#importing the keras packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#initialising the cnn
classifier = Sequential()

#convolutional layer #1

classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3), activation = "relu"))


#max pooling layer

classifier.add(MaxPooling2D(pool_size=(2,2)))

# convolution layer #2
classifier.add(Convolution2D(32,3,3,activation = "relu"))

#max pooling layer

classifier.add(MaxPooling2D(pool_size=(2,2)))


# flattening

classifier.add(Flatten())

# fully connected layer : classification

classifier.add(Dense(units = 128,kernel_initializer = "uniform", activation = "relu"))
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid" ))

classifier.compile(optimizer="adam",loss = "binary_crossentropy",metrics=["accuracy"])

# image preprocessing

# image augmentation

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',target_size=(64, 64), batch_size=32,class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',target_size=(64, 64), batch_size=32,class_mode='binary')

classifier.fit_generator(training_set,steps_per_epoch=250, epochs= 25, validation_data = test_set, validation_steps= 63)

#making new predictions
#
import numpy as np
from keras.preprocessing import image
test_image = image.load_img("dataset/single_prediction/cat.jpg",target_size=(64, 64)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
if result[0][0] == 1:
    prediction = "dog"
else:
    prediction = "cat"