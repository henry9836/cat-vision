#!/bin/python

import cv2
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras import layers 
import numpy as np
import pathlib
import random
import os
import sys

def testData(snakes, cucumbers, model, sizeY, sizeX, class_names):
    userInput = input("Test with new input [(i)nternet/(L)ocal test files]\nSystem Controls [(q)uit/(s)ave and Quit]\n [i/q/s/L]: ")
    img = ""
    img_path = ""

    # GET IMAGE
    if (userInput == "i"):
        url = input("Enter image url: ")
        cwd = os.getcwd()
        if os.path.exists(cwd + '/tmp_web'):
            os.remove(cwd + '/tmp_web')
        img_path = tf.keras.utils.get_file(cwd + '/tmp_web', origin=url)
    elif (userInput == "q"):
        sys.exit()
    elif (userInput == "s"):
        model.save('model_export.keras')
        sys.exit()
    else:
        if random.random() < .5:
            img_path = str(random.choice(snakes))
        else:
            img_path = str(random.choice(cucumbers))

    # LOAD THE IMAGE TO BE THE CORRECT SIZE
    img = tf.keras.utils.load_img(
        img_path, target_size=(sizeY, sizeX)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)

    score = tf.nn.softmax(predictions[0])

    # DISPLAY RESULT
    result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    shortResult = "{} {:.2f}%".format(class_names[np.argmax(score)], 100 * np.max(score))
    
    print(result)
    print(img_path)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    imgS = cv2.resize(img, (960, 540)) 
    imgS = cv2.putText(imgS, shortResult, (10, 75), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 8, cv2.LINE_AA) 
    imgS = cv2.putText(imgS, shortResult, (10, 75), cv2.FONT_HERSHEY_DUPLEX, 1, (28, 172, 255), 2, cv2.LINE_AA) 
    cv2.imshow(result, imgS)
    #cv2.moveWindow(result, 2160, 500)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def createModel():
    # DEFINE THE MODEL
    model = Sequential([
    layers.Rescaling(1./255, input_shape=(sizeY, sizeX, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
    ])

    # COMPILE THE MODEL
    model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

    # VIEW THE MODEL
    model.summary()

    #BEGIN TRAIN
    epochs=15
    history = model.fit(
        trainingDataset,
        validation_data=validationDataset,
        epochs=epochs
    )

    return model

sizeY = 640
sizeX = 640
batchSize = 32

TrainDir = "cat_vision/train/"
ValidationDir = "cat_vision/valid/"

train_data_dir = pathlib.Path(TrainDir).with_suffix('')
val_data_dir = pathlib.Path(TrainDir).with_suffix('')
testing_data_dir = pathlib.Path(TrainDir).with_suffix('')

snakes = list(testing_data_dir.glob('Snake/*'))
cucumbers = list(testing_data_dir.glob('Cucumber/*'))

trainingDataset = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="training",
    seed=69,
    image_size=(sizeX, sizeY),
    batch_size=batchSize
)

validationDataset = tf.keras.utils.image_dataset_from_directory(
    val_data_dir,
    validation_split=0.2,
    subset="validation",
    seed=69,
    image_size=(sizeX, sizeY),
    batch_size=batchSize
)

class_names = trainingDataset.class_names
num_classes = len(class_names)

# OPTMISE
AUTOTUNE = tf.data.AUTOTUNE

trainingDataset = trainingDataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validationDataset = validationDataset.cache().prefetch(buffer_size=AUTOTUNE)

model = []

cwd = os.getcwd()
if os.path.exists(cwd + '/model_export.keras'):
    userInput = input("Previous trained model found would you like to use it? [Y/n]: ")
    if (userInput == "n"):
        os.remove(cwd + '/model_export.keras')
        model = createModel()
    else:
        model = tf.keras.models.load_model('model_export.keras')
        model.summary()
else:
    model = createModel()

# TEST OUR DATASET!
snakes = list(testing_data_dir.glob('Snake/*'))
cucumbers = list(testing_data_dir.glob('Cucumber/*'))

while True:
    testData(snakes, cucumbers, model, sizeY, sizeX, class_names)

