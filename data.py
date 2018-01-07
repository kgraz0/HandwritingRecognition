import os
from PIL import Image
import numpy as np
from scipy import misc
import PIL.ImageOps

## save the name of the training and testing data set folders. This will get passed into
trainFolder = "train"
testFolder = "test"

def loadImages(folder):
    classes = list()
    imgCount = 0

    for dir in os.listdir(folder):
        classes.append(os.path.join(folder, dir))

    for eachClass in classes:
        for eachImage in os.listdir(eachClass):
            imageX = os.path.join(eachClass, eachImage)

            imgCount += 1

            if(imgCount % 200 == 0):
                print(str(imgCount) + " images processed.")

            img = Image.open(imageX)
            img = img.convert('L')

            newImage = otsuThreshold(img)
            newImage = PIL.ImageOps.invert(newImage)
            newImage.save(imageX)

def otsuThreshold(image):
    intensityArray = []
    for width in range (image.size[0]):
        for height in range (image.size[1]):
            pixelIntensity = image.getpixel((width, height))
            intensityArray.append(pixelIntensity)

    hist = np.histogram(intensityArray, range(0, 257))

    # calculate total pixels
    totalPixels = image.size[0]*image.size[1]
    current_max, threshold, sumT, sumF, sumB, weightB, weightF = 0, 0, 0, 0, 0, 0, 0
    for i in range(0, 256):
        sumT += i * hist[0][i]
    for i in range(0, 256):
        weightB += hist[0][i]
        weightF = totalPixels - weightB
        if weightF == 0:
            break
        sumB += i * hist[0][i]
        sumF = sumT - sumB
        meanB = sumB / weightB
        meanF = sumF / weightF
        varBetween = weightB * weightF
        varBetween *= (meanB - meanF) * (meanB - meanF)
        if varBetween > current_max:
            current_max = varBetween
            threshold = i
    thresholdedImage = applyThreshold(threshold, image)
    return thresholdedImage


def applyThreshold(t, image):
  intensity_array = []
  for w in range(0,image.size[1]):
    for h in range(0,image.size[0]):
      intensity = image.getpixel((h,w))
      if (intensity <= t):
        x = 0
      else:
        x = 255
      intensity_array.append(x)
  image.putdata(intensity_array)
  return image

def splitDataset(folder, trainingData):
    classes = list()
    images = list()

    for dir in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, dir)):
             classes.append(dir)

    for image in classes:
        images.append(os.listdir(os.path.join(folder, image)))

    if (trainingData == True):
        X, Y, lbl = loadNumpy(folder, images, classes)

        datasetSize = len(classes) * len(images[0])
        xTrain = X[:(datasetSize*8)//10]
        yTrain = Y[:(datasetSize*8)//10]
        xVal = X[(datasetSize*8)//10:]
        yVal = Y[(datasetSize*8)//10:]

        return xTrain, yTrain, xVal, yVal
    else:
        xTest, yTest, lbl = loadNumpy(folder, images, classes)
        return xTest, yTest, lbl

def loadNumpy(folder, images, classes):
    X = []
    Y = []

    for i in range(len(classes)):
        for j in range(len(images[0])):
            imageLocation = os.path.join(folder, classes[i], images[i][j])
            image = misc.imread(imageLocation)
            X.append(image)
            Y.append(i)
    X = np.array(X)
    Y = np.array(Y)
    labels = np.array(images)
    return X, Y, labels

#loadImages(trainFolder)
#loadImages(testFolder)
xTrain, yTrain, xVal, yVal = splitDataset(trainFolder, True)
xTest, yTest, hex = splitDataset(testFolder, False)

saver = open("splitDataset", "wb")
np.save(saver, xTrain)
np.save(saver, yTrain)
np.save(saver, xVal)
np.save(saver, yVal)
np.save(saver, xTest)
np.save(saver, yTest)
np.save(saver, hex)
saver.close()