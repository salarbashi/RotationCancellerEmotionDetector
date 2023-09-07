import tensorflow as tf

# set session limits
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.regularizers import l2


def AlexVgg(img_shape=(224, 224, 3), an_classes=10, vn_classes=8, weights=None):

    sharedDesne = Dense(4096)

    alexnet = alexnet_model(img_shape, an_classes, weights)

    vgg16 = ModifiedVGG16(img_shape, vn_classes)


    # completing alexnet
    alexnet.add(sharedDesne)
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))
    alexnet.add(Dense(an_classes))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('softmax'))

    # completing vgg16
    vgg16.add(sharedDesne)
    vgg16.add(Dense(vn_classes, activation='softmax', name='predictions'))

    return alexnet, vgg16


def alexnet_model(img_shape, n_classes, l2_reg=0.,
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
    # alexnet.add(Dense(4096))
    # alexnet.add(BatchNormalization())
    # alexnet.add(Activation('relu'))
    # alexnet.add(Dropout(0.5))
    #
    # # Layer 8
    # alexnet.add(Dense(n_classes))
    # alexnet.add(BatchNormalization())
    # alexnet.add(Activation('softmax'))

    if weights is not None:
        alexnet.load_weights(weights)

    return alexnet


def ModifiedVGG16(input_shape=(224, 224, 3),
                  classes=8):
    # Initialize model
    ModifiedVGG = Sequential()

    # Block 1
    ModifiedVGG.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=input_shape))
    # ModifiedVGG.add(BatchNormalization())
    ModifiedVGG.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    # ModifiedVGG.add(BatchNormalization())
    ModifiedVGG.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    # ModifiedVGG.add(BatchNormalization())

    # Block 2
    ModifiedVGG.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    # ModifiedVGG.add(BatchNormalization())
    ModifiedVGG.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    # ModifiedVGG.add(BatchNormalization())
    ModifiedVGG.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    # ModifiedVGG.add(BatchNormalization())

    # Block 3
    ModifiedVGG.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    # ModifiedVGG.add(BatchNormalization())
    ModifiedVGG.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    # ModifiedVGG.add(BatchNormalization())
    ModifiedVGG.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    # ModifiedVGG.add(BatchNormalization())
    ModifiedVGG.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    # ModifiedVGG.add(BatchNormalization())

    # Block 4
    ModifiedVGG.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    # ModifiedVGG.add(BatchNormalization())
    ModifiedVGG.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    # ModifiedVGG.add(BatchNormalization())
    ModifiedVGG.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    # ModifiedVGG.add(BatchNormalization())
    ModifiedVGG.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    # ModifiedVGG.add(BatchNormalization())

    # Block 5
    ModifiedVGG.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    # ModifiedVGG.add(BatchNormalization())
    ModifiedVGG.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    # ModifiedVGG.add(BatchNormalization())
    ModifiedVGG.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    ModifiedVGG.add(BatchNormalization())
    ModifiedVGG.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
    # ModifiedVGG.add(BatchNormalization())

    # FC
    ModifiedVGG.add(Flatten(name='flatten'))
    ModifiedVGG.add(Dense(3072, activation='relu', name='fc1'))
    # ModifiedVGG.add(Dense(4096, activation='relu', name='fc2'))
    # ModifiedVGG.add(Dense(classes, activation='softmax', name='predictions'))

    return ModifiedVGG
  
  

  
  

  
  
  
#connect to google drive
from google.colab import drive
drive.mount('/content/gdrive')
  
  
  
from tensorflow.python.keras import applications

from tensorflow.python.keras.optimizers import SGD, Adagrad, Adam, Adadelta, RMSprop, Adamax, Nadam
from tensorflow.python.keras.losses import mean_squared_error, categorical_crossentropy


from tensorflow.python.keras.models import Sequential
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image
import os, sys
import numpy as np
import math
import datetime

# emotionImagesDirectory = "/content/gdrive/My Drive/Colab Notebooks/EmotionExpressionPython/CKP4EmotionDetectGSedRotatedCroppedCircledImages"
emotionImagesDirectory = "/content/gdrive/My Drive/Colab Notebooks/EmotionExpressionPython/CKP4EmotionDetectFixedByNetAdadelta"
originalImagesDirectory = "/content/gdrive/My Drive/Colab Notebooks/EmotionExpressionPython/CKPFaceGSedRotatedCroppedCircledImages"
emotionsDirectory = "/content/gdrive/My Drive/Colab Notebooks/EmotionExpressionPython/CKPEmotions"
vggTrainingPercent = 80
alexnetTrainingPercent = 80
angleRatio = 10
vggTotalTrainingCount = int((len(os.listdir(emotionImagesDirectory))))
vggTrainingSetCount = int((len(os.listdir(emotionImagesDirectory)) * vggTrainingPercent / 100))
vggTestSetCount = vggTotalTrainingCount - vggTrainingSetCount
alexnetTotalTrainingCount = int((len(os.listdir(originalImagesDirectory))))
alexnetTrainingSetCount = int((len(os.listdir(originalImagesDirectory)) * alexnetTrainingPercent / 100))
alexnetTestSetCount = alexnetTotalTrainingCount - alexnetTrainingSetCount
numOfEmotions = 8
vEpochs = 40
aEpochs = 30
alr = 0.01
vlr = 0.009
aoptimizer = Adam(lr=alr)
voptimizer = Adadelta(lr=vlr)
# weights = 'imagenet'
weights = None
vgg16_batch_size = 32
vgg16_test_batch_size = 5
alexnet_batch_size = 25
alexnet_test_batch_size = 25
height = 100
width = 100
channels = 3


def DPLine(Text, n=1):
    for _ in range(n):
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')
    print(Text)


def getImgEmotion(fileAddress):
    result = 0
    try:
        file = open(fileAddress, "r")
        fileData = int(float(file.readline().lstrip().replace('\n', '')))
        result = fileData
    except:
        print("cannot get emotion from file {}".format(fileAddress))
        # os.remove(originalImagesDirectory+"/"+fileName)
    return result


def cnvrtEmotionFileName(filename: str):
    return filename.replace('_emotion', '')


def getImgEmotionsDic(EmotionDbDir):
    angleDic = dict()
    for file in os.listdir(EmotionDbDir):
        angleDic[cnvrtEmotionFileName(file)] = getImgEmotion(EmotionDbDir + "/" + file)
    return angleDic


def getImgEmotionsLst(EmotionDbDir):
    angleLst = list()
    for file in os.listdir(EmotionDbDir):
        angleLst.append(getImgEmotion(EmotionDbDir + "/" + file))
    return angleLst


def getNetInputFromDir(ImgDbDir, count=None):
    if count != None:
        SetCount = count
    else:
        SetCount = vggTrainingSetCount

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


def getPartOfNetInputFromDir(ImgDbDir, totalTrainingCount, min, max=None):
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


def getNetEmotionLabelsFromDir(EmotionDbDir):
    InputLabels = [cnvrtEmotion2NetOutput(int(x)) for x in getImgEmotionsLst(EmotionDbDir)]
    InputLabels = np.asanyarray(InputLabels)
    return InputLabels


def cnvrtAngle2NetOutput(angle):
    outputClassNumber = int(angle / angleRatio)
    result = np.zeros((int(360 / angleRatio)))
    result[outputClassNumber] = 1
    return result


def getImgAngle(fileName):
    temp = fileName.split(".")[0].split("_")[-1]
    try:
        int(temp)
    except:
        print(fileName)
        # os.remove(originalImagesDirectory+"/"+fileName)
    return temp


def getImgAnglesLst(ImgDbDir):
    angleLst = list()
    for file in os.listdir(ImgDbDir):
        angleLst.append(getImgAngle(file))
    return angleLst


def getAngleNetLabelsFromDir(ImgDbDir):
    InputLabels = [cnvrtAngle2NetOutput(int(x)) for x in getImgAnglesLst(ImgDbDir)]
    InputLabels = np.asanyarray(InputLabels)
    return InputLabels


def getSpecificImg(index, ImgDir):
    img = image.load_img(path=ImgDir + "/" + os.listdir(ImgDir)[index], color_mode="grayscale",
                         target_size=(height, width))
    img = image.img_to_array(img).reshape((height, width))
    img = np.stack((img,) * channels, axis=-1)
    return img.reshape((1, height, width, channels))


def getAlexNetInputFromDirGenerator(ImgDir, batch_size, training_set_count):
    if batch_size > training_set_count:
        raise ValueError('Batch size cannot be greater than training set count!')
    numOfBatches = math.ceil(training_set_count / batch_size)
    InputLabels = getAngleNetLabelsFromDir(ImgDir)
    while True:
        for i in range(numOfBatches):
            yield getPartOfNetInputFromDir(ImgDir, training_set_count, i * batch_size,
                                           i * batch_size + batch_size), InputLabels[
                                                                         i * batch_size:i * batch_size + batch_size]


def getVggNetInputFromDirGenerator(ImgDir, LabelDir, batch_size, trainingSetCount):
    if batch_size > trainingSetCount:
        raise ValueError('Batch size cannot be greater than training set count!')
    numOfBatches = math.ceil(trainingSetCount / batch_size)
    InputLabels = getNetEmotionLabelsFromDir(LabelDir)
    while True:
        for i in range(numOfBatches):
            yield getPartOfNetInputFromDir(ImgDir, trainingSetCount, i * batch_size,
                                           i * batch_size + batch_size), InputLabels[
                                                                         i * batch_size:i * batch_size + batch_size]


def getAlexNetTestInputFromDirGenerator(ImgDir, batch_size, test_set_count, total_training_count):
    if batch_size > test_set_count:
        raise ValueError('Batch size cannot be greater than test set count!')
    numOfBatches = math.ceil(test_set_count / batch_size)
    InputLabels = getAngleNetLabelsFromDir(ImgDir)
    trainingSetCount = total_training_count - test_set_count
    # while True:
    for i in range(numOfBatches):
        if i == numOfBatches - 1:
            yield getPartOfNetInputFromDir(ImgDir, alexnetTrainingSetCount, i * batch_size + trainingSetCount,
                                           total_training_count), InputLabels[
                                                                  i * batch_size + trainingSetCount:total_training_count]
        else:
            yield getPartOfNetInputFromDir(ImgDir, alexnetTrainingSetCount, i * batch_size + trainingSetCount,
                                           i * batch_size + batch_size + trainingSetCount), InputLabels[
                                                                                            i * batch_size + trainingSetCount:i * batch_size + batch_size + trainingSetCount]


def getVggNetTestInputFromDirGenerator(ImgDir, LabelsDir, batch_size, test_set_count,
                                       total_training_count):
    if batch_size > test_set_count:
        raise ValueError('Batch size cannot be greater than test set count!')
    numOfBatches = math.ceil(test_set_count / batch_size)
    InputLabels = getNetEmotionLabelsFromDir(LabelsDir)
    trainingSetCount = total_training_count - test_set_count
    # while True:
    for i in range(numOfBatches):
        if i == numOfBatches - 1:
            yield getPartOfNetInputFromDir(ImgDir, total_training_count, i * batch_size + trainingSetCount,
                                           total_training_count), InputLabels[
                                                                  i * batch_size + trainingSetCount:total_training_count]
        else:
            yield getPartOfNetInputFromDir(ImgDir, total_training_count, i * batch_size + trainingSetCount,
                                           i * batch_size + batch_size + trainingSetCount), InputLabels[
                                                                                            i * batch_size + trainingSetCount:i * batch_size + batch_size + trainingSetCount]


def getTestLabels(EmotionDbDir):
    Result = getNetEmotionLabelsFromDir(EmotionDbDir)[vggTrainingSetCount:vggTotalTrainingCount]
    Result = [cnvrtNetOutput2Emotion(x) for x in Result]
    return Result


def cnvrtEmotion2NetOutput(emotion: int):
    result = np.zeros(numOfEmotions)
    result[emotion] = 1
    return result


def cnvrtNetOutput2Emotion(outputArray):
    maxIndex = outputArray.argmax()
    return maxIndex


def evaluateNetPredictByGenerator(net: Sequential, Generator, test_set_count):
    CorrectCount, IncorrectCount = 0, 0
    try:
        while True:
            item = next(Generator)
            predict = np.asanyarray([cnvrtNetOutput2Emotion(x) for x in net.predict(item[0])])
            netOut = np.asanyarray([cnvrtNetOutput2Emotion(x) for x in item[1]])
            CorrectCount += np.sum(predict == netOut)
            IncorrectCount += np.sum(predict != netOut)
            DPLine("{}/{}, Correct:{}, Incorrect:{}".format(CorrectCount + IncorrectCount, test_set_count,
                                                            CorrectCount, IncorrectCount))

    except StopIteration:
        return CorrectCount, IncorrectCount


print(tf.__version__)

print("Define Net...")
alexnet = AlexVgg(img_shape=(height, width, channels), an_classes=int(360 / angleRatio), vn_classes=numOfEmotions, weights=weights)[0]

# print("Loading weights...")
# net.load_weights("/content/gdrive/My Drive/Colab Notebooks/weights.h5")

print("Compile Net...")
alexnet.compile(optimizer=aoptimizer, loss=categorical_crossentropy, metrics=["accuracy"])
# vgg16.compile(optimizer=voptimizer, loss=categorical_crossentropy, metrics=["accuracy"])
print("AlexNet summary...")
print(alexnet.summary())
print("Vgg16 summary...")
# print(vgg16.summary())


print("AlexNet train and test...")
print("*************************************")
print("Preparing Train data...")
print("Train set Count:{}".format(alexnetTrainingSetCount))
aGenerator = getAlexNetInputFromDirGenerator(originalImagesDirectory, alexnet_batch_size, alexnetTrainingSetCount)

print("Fit by Generator...")
fitStartTime = datetime.datetime.now()
alexnet.fit_generator(aGenerator, steps_per_epoch=int(alexnetTrainingSetCount / alexnet_batch_size), epochs=aEpochs)
fitStopTime = datetime.datetime.now()
print("After Fit...")

print("Saving weights...")
# alexnet.save_weights(
#     "/content/gdrive/My Drive/Colab Notebooks/weights_" + str(vEpochs) + "_" + str(
#         voptimizer.__class__.__name__) + ".h5")

print("Preparing Tests data...")
print("Test set Count:{}".format(alexnetTestSetCount))
AlexNetTestInputGenerator = getAlexNetTestInputFromDirGenerator(originalImagesDirectory, alexnet_batch_size,
                                                                alexnetTestSetCount, alexnetTotalTrainingCount)

print("Evaluating net:")
evaluateStartTime = datetime.datetime.now()
Result = evaluateNetPredictByGenerator(alexnet, AlexNetTestInputGenerator, alexnetTestSetCount)
evaluateStopTime = datetime.datetime.now()
print("Source:{} / Epochs:{} /Optimizer:{} /lr:{} /Weights:{} /Correct:{} / Incorrect:{} / Percent:{}%".format(originalImagesDirectory.split('/')[-1], str(aEpochs), str(aoptimizer.__class__.__name__), alr, weights, Result[0], Result[1], Result[0] / alexnetTestSetCount * 100))
open("/content/gdrive/My Drive/Colab Notebooks/Results.txt", "a").write(
    "Source:{} / Epochs:{} /Optimizer:{} /lr:{} /Weights:{} /Correct:{} / Incorrect:{} / Percent:{}%".format(originalImagesDirectory.split('/')[-1], str(aEpochs), str(aoptimizer.__class__.__name__), alr, weights, Result[0], Result[1], Result[0] / alexnetTestSetCount * 100))

print("Evaluate time:{}".format(evaluateStopTime - evaluateStartTime))
open("/content/gdrive/My Drive/Colab Notebooks/Results.txt", "a").write(
    "Fit time:{} / Evaluate time:{}\n".format(fitStopTime - fitStartTime, evaluateStopTime - evaluateStartTime))

print("Vgg16 train and test...")
print("*************************************")
print("Preparing Train data...")
print("Train set Count:{}".format(vggTrainingSetCount))
vGenerator = getVggNetInputFromDirGenerator(emotionImagesDirectory, emotionsDirectory, vgg16_batch_size,
                                            vggTrainingSetCount)

print("Fit by Generator...")
fitStartTime = datetime.datetime.now()
vgg16.fit_generator(vGenerator, steps_per_epoch=int(vggTrainingSetCount / vgg16_batch_size), epochs=vEpochs)
fitStopTime = datetime.datetime.now()
print("After Fit...")

# print("Saving weights...")
# SVMnet.save_weights(
#     "/content/gdrive/My Drive/Colab Notebooks/weights_" + str(epochs) + "_" + str(optimizer.__class__.__name__) + ".h5")

print("Preparing Tests data...")
print("Test set Count:{}".format(vggTestSetCount))
VGGTestInputGenerator = getVggNetTestInputFromDirGenerator(emotionImagesDirectory, emotionsDirectory,
                                                           vgg16_test_batch_size, vggTestSetCount,
                                                           vggTotalTrainingCount)

print("Evaluating net:")
evaluateStartTime = datetime.datetime.now()
Result = evaluateNetPredictByGenerator(vgg16, VGGTestInputGenerator, vggTestSetCount)
evaluateStopTime = datetime.datetime.now()
print("Source:{} / Epochs:{} /Optimizer:{} /lr:{} /Weights:{} /Correct:{} / Incorrect:{} / Percent:{}%".format(emotionImagesDirectory.split('/')[-1], str(vEpochs), str(voptimizer.__class__.__name__), vlr, weights, Result[0], Result[1], Result[0] / vggTestSetCount * 100))
open("/content/gdrive/My Drive/Colab Notebooks/Results.txt", "a").write(
    "Source:{} / Epochs:{} /Optimizer:{} /lr:{} /Weights:{} /Correct:{} / Incorrect:{} / Percent:{}%".format(emotionImagesDirectory.split('/')[-1], str(vEpochs), str(voptimizer.__class__.__name__), vlr, weights, Result[0], Result[1], Result[0] / vggTestSetCount * 100))

print("Evaluate time:{}".format(evaluateStopTime - evaluateStartTime))
open("/content/gdrive/My Drive/Colab Notebooks/Results.txt", "a").write(
    "Fit time:{} / Evaluate time:{}\n".format(fitStopTime - fitStartTime, evaluateStopTime - evaluateStartTime))

print("Releasing memory...")
tf.reset_default_graph()
print("End!!")
