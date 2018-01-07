Currently non-functional version as the training and testing data file is too large to be uploaded.

## If you want to add your own data:
-> you can create two folders called train and test within the same directory
-> Within these folders, have an individual folder for each alphabetic letter
-> 460 examples of a single alphabet character as testing data in .png format
-> 1840 examples of a single alphabet character as training data in .png format

You will need to run the data.py file to allow the pre-processing algorithm to run before training the model. This process will include application of otsu thresholding to make sure that the images only have black and white colours for easier processing  After the process is done, splitDataset file will appear in the directory so that you do not need to perform this process again.

You will now need to run model.py file for the convolutional neural network to be trained based on the data previously processed. This training method utilizes the use of Keras library to run 10 epochs to train the data. When the process is done, more files will appear in the directory -- these files form the model by saving the trained weights so you do not have to train it again every time you want to run it.

Currently the work on the interface.py file is not finished, therefore the trained model cannot yet be utilized to give reading on unseen data.

However, if you want to test the final result, you can run interface.py file, select an image file to be processed and the pytesseract library will attempt to extract the words from the image and then give a dictionary listing for that word (if it exists).

## Future work:
-> Optimize the convolutional neural network
-> Test the CNN on unseen data once finished
-> Add more data & more variety of data to increase accuracy
-> Implement more pre-processing methods to eliminate as much noise as possible and improve results
-> Feedback system -- allowing the user to give a score of accuracy, feeding the result back into the network
-> Upper-case and lower-case characters, digits, etc
-> Real-time recognition, potential to be moved onto mobile platforms