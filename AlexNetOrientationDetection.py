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
#     alexnet.add(Dense(4096))
#     alexnet.add(BatchNormalization())
#     alexnet.add(Activation('relu'))
#     alexnet.add(Dropout(0.5))

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


# import AlexNet
from tensorflow.python.keras.optimizers import SGD, Adagrad, Adam, Adadelta, RMSprop, Adamax, Nadam
from tensorflow.python.keras.losses import mean_squared_error, categorical_crossentropy
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.datasets import mnist
import numpy as np, sys
import math
import datetime

# This address identifies the TPU we'll use when configuring TensorFlow.
# TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']

originalImagesDirectory = "/content/gdrive/My Drive/Colab Notebooks/EmotionExpressionPython/CKPFaceGSedRotatedCroppedCircledImages"
trainingPrecent = 90
totalTrainingCount = int((len(os.listdir(originalImagesDirectory))))
trainingSetCount = int((len(os.listdir(originalImagesDirectory)) * trainingPrecent / 100))
testSetCount = totalTrainingCount - trainingSetCount
angleRatio = 10
epochs = 30
lr = 0.01
optimizer = Nadam(lr=lr)
batch_size = 32
test_batch_size = 100
height= 100
width = 100
channels = 1





def DPLine(Text, n=1):
    for _ in range(n):
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')
    print(Text)
    
    
def getImgAngle(fileName):
    temp = fileName.split(".")[0].split("_")[-1]
    try:
        int(temp)
    except:
        print(fileName)
        # os.remove(originalImagesDirectory+"/"+fileName)
    return temp


def getImgAnglesDic(ImgDbDir):
    angleDic = dict()
    for file in os.listdir(ImgDbDir):
        angleDic[file] = getImgAngle(file)
    return angleDic


def getImgAnglesLst(ImgDbDir):
    angleLst = list()
    for file in os.listdir(ImgDbDir):
        angleLst.append(getImgAngle(file))
    return angleLst


def getNetInputFromDir(ImgDbDir, count=None):
    if count != None:
        SetCount = count
    else:
        SetCount = trainingSetCount

    inputArray = np.zeros((SetCount, height, width, channels))

    current = 0
    for file in os.listdir(ImgDbDir):
        if current == SetCount:
            break

        img = image.load_img(path=ImgDbDir + "/" + file, color_mode="grayscale",
                             target_size=(height, width, channels))
        img = image.img_to_array(img)
        inputArray[current] = img

        current += 1
    return inputArray


def getPartOfNetInputFromDir(ImgDbDir, min, max=None):
    SetCount = totalTrainingCount

    if max == None:
        max = totalTrainingCount

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


def getNetLabelsFromDir(ImgDbDir):
    InputLabels = [cnvrtAngle2NetOutput(int(x)) for x in getImgAnglesLst(ImgDbDir)]
    InputLabels = np.asanyarray(InputLabels)
    return InputLabels


def getSpecificImg(index, ImgDir):
    img = image.load_img(path=ImgDir + "/" + os.listdir(ImgDir)[index], grayscale=True,
                         target_size=(height, width, channels))
    img = image.img_to_array(img)
    return img.reshape((1, height, width, channels))


def getNetInputFromDirGenerator(ImgDir, batch_size):
    if batch_size > trainingSetCount:
        raise ValueError('Batch size cannot be greater than training set count!')
    numOfBatches = math.ceil(trainingSetCount / batch_size)
    InputLabels = getNetLabelsFromDir(ImgDir)
    while True:
        for i in range(numOfBatches):
            yield getPartOfNetInputFromDir(ImgDir, i * batch_size, i * batch_size + batch_size), InputLabels[
                                                                                                 i * batch_size:i * batch_size + batch_size]
            
def getNetTestInputFromDirGenerator(ImgDir, batch_size):
    if batch_size > testSetCount:
        raise ValueError('Batch size cannot be greater than test set count!')
    numOfBatches = math.ceil(testSetCount / batch_size)
    InputLabels = getNetLabelsFromDir(ImgDir)
    # while True:
    for i in range(numOfBatches):
        if i == numOfBatches - 1:
            yield getPartOfNetInputFromDir(ImgDir, i * batch_size + trainingSetCount,
                                           totalTrainingCount), InputLabels[
                                                                i * batch_size + trainingSetCount:totalTrainingCount]
        else:
            yield getPartOfNetInputFromDir(ImgDir, i * batch_size + trainingSetCount,
                                           i * batch_size + batch_size + trainingSetCount), InputLabels[
                                                                                            i * batch_size + trainingSetCount:i * batch_size + batch_size + trainingSetCount]
            



def getTestLabels():
    Result = getNetLabelsFromDir(originalImagesDirectory)[trainingSetCount:totalTrainingCount]
    Result = [cnvrtNetOutput2Angle(x) for x in Result]
    return Result
            
            
def cnvrtAngle2NetOutput(angle):
    outputClassNumber = int(angle / angleRatio)
    result = np.zeros((int(360 / angleRatio)))
    result[outputClassNumber] = 1
    return result


def cnvrtNetOutput2Angle(outputArray):
    maxIndex = outputArray.argmax()
    return maxIndex * angleRatio
  
  
def evaluateNetPredict(net: Sequential, Inputs, Labels, batchSize=batch_size):
    if len(Inputs) != len(Labels):
        raise ValueError('the size of Input and Labels of predict evaluate cannot be of different size')
    Predicts = net.predict(Inputs, batch_size=batchSize)
    Predicts = [cnvrtNetOutput2Angle(x) for x in Predicts]
    return np.sum(Labels == Predicts)
  
  
def evaluateNetPredictByGenerator(net: Sequential, Generator, batchSize=batch_size):
    CorrectCount, IncorrectCount = 0, 0
    try:
        while True:
            item = next(Generator)
            predict = np.asanyarray([cnvrtNetOutput2Angle(x) for x in net.predict(item[0])])
            netOut = np.asanyarray([cnvrtNetOutput2Angle(x) for x in item[1]])
            CorrectCount += np.sum(predict == netOut)
            IncorrectCount += np.sum(predict != netOut)
            DPLine("{}/{}, Correct:{}, Incorrect:{}".format(CorrectCount + IncorrectCount, testSetCount,
                                                            CorrectCount, IncorrectCount))
            # if predict == cnvrtNetOutput2Angle(item[1]):
            #     CorrectCount += 1
            #     print("{}/{}, Correct:{}, Incorrect:{}".format(CorrectCount + IncorrectCount, testSetCount,
            #                                                      CorrectCount, IncorrectCount), end="\r")
            # else:
            #     IncorrectCount += 1
            #     print("{}/{}, Correct:{}, Incorrect:{}".format(CorrectCount + IncorrectCount, testSetCount,
            #                                                      CorrectCount, IncorrectCount), end="\r")
    except StopIteration:
        return CorrectCount, IncorrectCount
  
  
def evaluateNetByGenerator(net: Sequential, InputGen, batchSize=batch_size):
    loss, acc = net.evaluate_generator(InputGen, steps=batch_size)
    return loss, acc

  


print(tf.__version__)


print("Define Net...")
net = alexnet_model((height, width, channels), int(360 / angleRatio))

# print("Loading weights...")
# net.load_weights("/content/gdrive/My Drive/Colab Notebooks/alexnet_weights_30_Adadelta.h5")
# sharedLayerWeights = np.load("/content/gdrive/My Drive/Colab Notebooks/vgg16layerfc2weights.npy")
# net.layers[27].set_weights(sharedLayerWeights)


print("Compile Net...")
net.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=["accuracy"])

print("Preparing Train data...")
print("Train set Count:{}".format(trainingSetCount))
generator = getNetInputFromDirGenerator(originalImagesDirectory, batch_size)

print("Fit by Generator...")
fitStartTime = datetime.datetime.now()
net.fit_generator(generator, steps_per_epoch= int(trainingSetCount / batch_size), epochs=epochs)
fitStopTime = datetime.datetime.now()
print("After Fit...")

# print("Saving weights...")
# net.save_weights("/content/gdrive/My Drive/Colab Notebooks/weights_"+str(epochs)+"_"+str(optimizer.__class__.__name__)+".h5")
# np.save("/content/gdrive/My Drive/Colab Notebooks/alexnetlayer27weights.npy", net.layers[27].get_weights())

print("Preparing Tests data...")
print("Test set Count:{}".format(testSetCount))
TestInputGenerator = getNetTestInputFromDirGenerator(originalImagesDirectory, test_batch_size)

print("Evaluating net:")
evaluateStartTime = datetime.datetime.now()
# Result = evaluateNetByGenerator(net, TestInputGenerator)
# print("Loss:{} / Acc:{}".format(Result[0], Result[1]))
Result = evaluateNetPredictByGenerator(net, TestInputGenerator)
evaluateStopTime = datetime.datetime.now()
print("Correct:{} / Incorrect:{} / Percent:{}%".format(Result[0], Result[1], Result[0] / testSetCount*100))
open("/content/gdrive/My Drive/Colab Notebooks/Results.txt", "a").write("Correct:{} / Incorrect:{} / Percent:{}%\n".format(Result[0], Result[1], Result[0] / testSetCount*100))

print("Fit time:{} / Evaluate time:{}".format(fitStopTime - fitStartTime, evaluateStopTime - evaluateStartTime))
open("/content/gdrive/My Drive/Colab Notebooks/Results.txt", "a").write("Fit time:{} / Evaluate time:{}\n".format(fitStopTime - fitStartTime, evaluateStopTime - evaluateStartTime))

print("Releasing memory...")
tf.reset_default_graph()
print("End!!")