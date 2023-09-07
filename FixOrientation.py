# Import necessary packages
import argparse
import tensorflow as tf

# set session limits
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# Import necessary components to build LeNet
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.regularizers import l2



def alexnet_model(img_shape=(224, 224, 3), n_classes=10, l2_reg=0.,
                  weights=None):
    # Initialize model
    alexnet = Sequential()

    # Layer 1
    alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,
                       padding='same', kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    alexnet.add(Conv2D(256, (5, 5), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(512, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 4
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # Layer 5
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(3072))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 7
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 8
    alexnet.add(Dense(n_classes))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('softmax'))

    if weights is not None:
        alexnet.load_weights(weights)

    return alexnet


def parse_args():
    """
	Parse command line arguments.
	Parameters:
		None
	Returns:
		parser arguments
	"""
    parser = argparse.ArgumentParser(description='AlexNet model')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional.add_argument('--print_model',
                          dest='print_model',
                          help='Print AlexNet model',
                          action='store_true')
    parser._action_groups.append(optional)
    return parser.parse_args()


















import sys, os

#connect to google drive
from google.colab import drive
drive.mount('/content/gdrive')


from tensorflow.python.keras.optimizers import SGD, Adagrad, Adam, Adadelta, RMSprop, Adamax, Nadam
from tensorflow.python.keras.losses import mean_squared_error, categorical_crossentropy
import os, math
import numpy as np
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Sequential
from PIL import Image
import datetime

originalImagesDirectory = "/content/gdrive/My Drive/Colab Notebooks/EmotionExpressionPython/CKPFaceGSedRotatedCroppedCircledImages"

totalImagesCount = int((len(os.listdir(originalImagesDirectory))))
height = 100
width = 100
channels = 1
angleRatio = 10
lr = 0.01
optimizer = Adadelta(lr=lr)

fixedImageDirectory = "/content/gdrive/My Drive/Colab Notebooks/EmotionExpressionPython/CKPFixedByNet" + str(optimizer.__class__.__name__)
netWeightDirectory = "/content/gdrive/My Drive/Colab Notebooks/weights_30_Adadelta.h5"

if not os.path.exists(fixedImageDirectory):
    os.mkdir(fixedImageDirectory)


def getFixedFileName(filename, predictedAngle):
    predictedAngle = int(predictedAngle)
    angle = filename.split(".")[0].split("_")[-1]
    usSeperated = filename.split(".")[0].split("_")
    dSeperated = filename.split(".")
    try:
        fixed = int(angle) - predictedAngle
        returning = dSeperated[0][:-len(usSeperated[-1])] + str(fixed) + "." + dSeperated[-1]
        return returning
    except:
        print(filename)


def getPartOfNetInputFromDir(ImgDbDir, min, max=None):
    SetCount = totalImagesCount

    if max == None:
        max = totalImagesCount

    inputArray = np.zeros((max - min, height, width, channels))

    current = 0
    for file in os.listdir(ImgDbDir)[min:max]:
        if current == SetCount:
            break

        img = image.load_img(path=ImgDbDir + "/" + file, color_mode="grayscale",
                             target_size=(height, width, channels))
        img = image.img_to_array(img)
        inputArray[current] = img

        current += 1
    return inputArray


def getSpecificImg(index, ImgDir):
    img = image.load_img(path=ImgDir + "/" + os.listdir(ImgDir)[index], grayscale=True,
                         target_size=(height, width, channels))
    img = image.img_to_array(img)
    return img.reshape((1, height, width, channels))


def getImgFromDir(filename, ImgDir):
    img = image.load_img(path=ImgDir + "/" + filename, grayscale=True,
                         target_size=(height, width, channels))
    img = image.img_to_array(img)
    return img.reshape((1, height, width, channels))


def getNetInputFromDirGenerator(ImgDir, batch_size):
    if batch_size > totalImagesCount:
        raise ValueError('Batch size cannot be greater than training set count!')
    numOfBatches = math.ceil(totalImagesCount / batch_size)
    for i in range(numOfBatches):
        if i == numOfBatches - 1:
            yield getPartOfNetInputFromDir(ImgDir, i * batch_size, totalImagesCount)
        else:
            yield getPartOfNetInputFromDir(ImgDir, i * batch_size, i * batch_size + batch_size)


def cnvrtAngle2NetOutput(angle):
    outputClassNumber = int(angle / angleRatio)
    result = np.zeros((int(360 / angleRatio)))
    result[outputClassNumber] = 1
    return result


def cnvrtNetOutput2Angle(outputArray):
    maxIndex = outputArray.argmax()
    return maxIndex * angleRatio


def predictImageAngle(net: Sequential, ImgDir, filename):
    return cnvrtNetOutput2Angle(net.predict(getImgFromDir(filename, ImgDir)))


def fixImages(ImgDir, net: Sequential):
    current = 0
    try:
        for file in os.listdir(ImgDir):
            predictedAngle = predictImageAngle(net, ImgDir, file)
            im = Image.open(ImgDir + "/" + file)
            im.rotate(predictedAngle).save(
                fixedImageDirectory + '/' + getFixedFileName(file, predictedAngle))
            print("{} percent completed! {}/{}".format((current + 1) / totalImagesCount * 100, current + 1,
                                                       totalImagesCount))
            current += 1
    except StopIteration:
        return
      
      
def getNumberOfUnfixed(ImgDir):
    count = 0
    for file in os.listdir(ImgDir):
        try:
            if int(file.split(".")[0].split("_")[-1]) != 0:
                count += 1
                print(file)
        except:
            None

    return count


print("Define Net...")
net = alexnet_model((height, width, channels), int(360 / angleRatio))

print("Loading weights...")
net.load_weights(netWeightDirectory)

print("Compile Net...")
net.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=["accuracy"])

print("Fixing Images Orientation ...")
fixStartTime = datetime.datetime.now()
fixImages(originalImagesDirectory, net)
fixStopTime = datetime.datetime.now()


print("Number of unfixed images:{}".format(getNumberOfUnfixed(fixedImageDirectory)))

print("Fix time:{}".format(fixStopTime - fixStartTime))

print("Releasing memory...")
tf.reset_default_graph()
print("End!")