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

# Get images
X = []
Y = []
for filename in os.listdir('./Train/'):
    inp = img_to_array(load_img('./Train/'+filename))
    inp = np.array(inp, dtype=float)
    inp = rgb2lab(1.0/255*inp)[:,:,0]
    inp = inp.reshape(1,32,32,1)
    X.append(inp)
    out = img_to_array(load_img('./Train/'+filename))
    out = np.array(out, dtype=float)
    out = rgb2lab(1.0 / 255 * out)[:, :, 1:]
    out /= 128
    out = out.reshape(1,32,32,2)
    Y.append(out)
X = np.array(X, dtype=float)
Y = np.array(Y, dtype=float)

# image = img_to_array(load_img('./Train/11746276_de3dec8201.jpg'))
# image = np.array(image, dtype=float)

# X = rgb2lab(1.0/255*image)[:,:,0]
# Y = rgb2lab(1.0/255*image)[:,:,1:]
# Y /= 128
# X = X.reshape(1, 32, 32, 1)
# Y = Y.reshape(1, 32, 32, 2)

# Building the neural network
model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

# Finish model
model.compile(optimizer='adam', loss='mse')

# Train Model
def generator(features, labels, batch_size):
 batch_features = np.zeros((batch_size, 32, 32, 1))
 batch_labels = np.zeros((batch_size, 32, 32, 2))
 while True:
   for i in range(batch_size):
     index= i
     batch_features[i] = features[index]
     batch_labels[i] = labels[index]
   yield batch_features, batch_labels

model.fit_generator(generator(X, Y, 150), epochs=2000, steps_per_epoch=1, verbose=1)

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("./Models/color_tensorflow_end.h5")

print('Finished to train dataset.')

# # Load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
#
# # load weights into new model
# loaded_model.load_weights("./Models/color_tensorflow_end.h5")
# print("Loaded model from disk..")
#
# loaded_model.compile(optimizer='adam', loss='mse')

# image = img_to_array(load_img('./Test/image.jpg'))
# image = Image.open('./Test/image.jpg')
# image = image.resize((96,96))

# image = img_to_array(load_img('./Images/image_2.jpg'))
# image = np.array(image, dtype=float)
#
# X = rgb2lab(1.0/255*image)[:,:,0]
# Y = rgb2lab(1.0/255*image)[:,:,1:]
# Y /= 128
# X = X.reshape(1, 32, 32, 1)
# Y = Y.reshape(1, 32, 32, 2)
#
# print(model.evaluate(X, Y, batch_size=1))
# output = model.predict(X)
# output *= 128

# # Output colorizations
# cur = np.zeros((32, 32, 3))
# cur[:,:,0] = X[0][:,:,0]
# cur[:,:,1:] = output[0]
# imsave("./Results/img_result.png", lab2rgb(cur))
# imsave("./result/img_gray_version.png", rgb2gray(lab2rgb(cur)))