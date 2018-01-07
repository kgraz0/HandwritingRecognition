from tkinter import *
from tkinter import filedialog
from PyDictionary import PyDictionary

from PIL import Image
import pytesseract
from pytesseract import image_to_string
import PIL.ImageOps

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'


dictionary = PyDictionary()

root = Tk()
root.withdraw()
filename = filedialog.askopenfilename()

image = Image.open(filename)
image = image.convert('L')
newImage = PIL.ImageOps.invert(image)
extractedWord = image_to_string(newImage, config='-psm 8')

print("Word Found: " + str(extractedWord))
print (dictionary.meaning(extractedWord))