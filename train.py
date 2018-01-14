from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose,UpSampling3D,Conv3D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb

# Parameters of Model
Batch_size = 20
Epochs = 50

# Get images
X = []
Y = []
for filename in os.listdir('./Train32/'):
    inp = img_to_array(load_img('./Train32/'+filename))
    inp = np.array(inp, dtype=float)
    inp = rgb2lab(1.0/255*inp)[:,:,0]
    inp = inp.reshape(1,32,32,1)
    X.append(inp)
    out = img_to_array(load_img('./Train96/'+filename))
    out = np.array(out, dtype=float)
    out = rgb2lab(1.0 / 255 * out)[:, :, 1:]
    out /= 128
    out = out.reshape(1,96,96,2)
    Y.append(out)
X = np.array(X, dtype=float)
Y = np.array(Y, dtype=float)

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
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((3, 3)))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

# Finish model
model.compile(optimizer='adam', loss='mse')
print(model.output)

# Train96 Model
def generator(features,labels, batch_size):
 batch_features = np.zeros((batch_size, 32, 32, 1))
 batch_labels = np.zeros((batch_size, 96, 96, 2))
 while True:
   for i in range(batch_size):
     index= i
     batch_features[i] = features[index]
     batch_labels[i] = labels[index]
   yield batch_features, batch_labels

# Train Model - learning rate inside 'adam' optimizer
model.fit_generator(generator(X ,Y,Batch_size), epochs=Epochs, steps_per_epoch=1, verbose=1)

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("./Models/color_tensorflow_end.h5")

print('Finished to train dataset.')

# Print Params & write to Log file
print('---------------------')
print('Model - Parameters:')
print('Batch Size: ', Batch_size)
print('Number of epoches: ', Epochs)
print('---------------------')

# Write parameters to a local file
with open('./Logs/params.txt', 'a') as the_file:
    the_file.write('Train:'  + '\n')
    the_file.write('Batch Size: ' +  str(batch_size) + '\n')
    the_file.write('Number of epoches: ' + str(Epochs) + '\n')
    the_file.write('----------------\n')