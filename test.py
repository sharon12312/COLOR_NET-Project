from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from skimage.transform import resize
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.models import model_from_json
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import numpy as np
import os
from PIL import Image
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
import random
import tensorflow as tf

# Load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("./Models/color_tensorflow_end.h5")
print("Loaded model from disk..")

loaded_model.compile(optimizer='adam', loss='mse')

# image = img_to_array(load_img('./Test/image.jpg'))
# image = Image.open('./Test/image.jpg')
# image = image.resize((96,96))

image = img_to_array(load_img('./Images/image_3.jpg'))
image = np.array(image, dtype=float)

X = rgb2lab(1.0/255*image)[:,:,0]
Y = rgb2lab(1.0/255*image)[:,:,1:]
Y /= 128
X = X.reshape(1, 32, 32, 1)
Y = Y.reshape(1, 32, 32, 2)

print(loaded_model.evaluate(X, Y, batch_size=1))
output = loaded_model.predict(X)
output *= 128

# Output colorizations
cur = np.zeros((32, 32, 3))
cur[:,:,0] = X[0][:,:,0]
cur[:,:,1:] = output[0]
imsave("./Results/img_result.png", lab2rgb(cur))
# imsave("./result/img_gray_version.png", rgb2gray(lab2rgb(cur)))