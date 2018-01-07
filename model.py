import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

saver = open("splitDataset", "rb")
xTrain = np.load(saver)
yTrain = np.load(saver)
xVal = np.load(saver)
yVal = np.load(saver)
xTest = np.load(saver)
yTest = np.load(saver)
labelNames = np.load(saver)
saver.close()

numPixels = xTrain.shape[1]*xTrain.shape[2]
xTrain = np.reshape(xTrain, (xTrain.shape[0], 1, xTrain.shape[1], xTrain.shape[2])).astype('float32')
xVal = np.reshape(xVal, (xVal.shape[0], 1, xVal.shape[1], xVal.shape[2])).astype('float32')
xTest = np.reshape(xTest, (xTest.shape[0], 1, xTest.shape[1], xTest.shape[2])).astype('float32')

xTrain = xTrain / 255
xVal = xVal / 255
xTest = xTest / 255

numClasses = len(labelNames)
yTrain = np_utils.to_categorical(yTrain, numClasses)
yVal = np_utils.to_categorical(yVal, numClasses)
yTest = np_utils.to_categorical(yTest, numClasses)

def convolutionalNetwork():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), activation='relu', input_shape=(1, 128, 128), padding='same'))
    model.add(MaxPooling2D(data_format='channels_first', pool_size=(2,2)))

    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(numClasses, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = convolutionalNetwork()

model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=10, batch_size=200, verbose=2)

scores = model.evaluate(xTest, yTest, verbose=0)
print(scores)

jsonModel = model.to_json()
f = open('savedModel.json', 'w')
f.write(jsonModel)
f.close()
model.save_weights('modelWeights.h5', overwrite=True)