
# coding: utf-8

# In[9]:


import csv
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from random import shuffle
import matplotlib.image as mpimg


# In[10]:


samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

print(samples.pop(0))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print("done")


# In[11]:


def invertr(Xtrain, ytrain):
    for i,j in zip(Xtrain, ytrain):
        temp = np.fliplr(i)
        Xtrain = np.concatenate((Xtrain, [temp]), axis = 0)
        temp2 = -1*j
        ytrain = np.concatenate((ytrain, [temp2]), axis = 0)
    return Xtrain, ytrain


# In[12]:


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = mpimg.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
 
                name = 'data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = mpimg.imread(name)
                left_angle = 0 if center_angle == 0 else (center_angle + 0.1 )
                images.append(left_image)
                angles.append(left_angle)
        
                name = 'data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = mpimg.imread(name)
                right_angle = 0 if center_angle == 0 else (center_angle - 0.1 )
                images.append(right_image)
                angles.append(right_angle)
           
            
            X_train = np.array(images)
            y_train = np.array(angles)
            
            X_train, y_train = invertr(X_train, y_train)
            
            
            
            yield sklearn.utils.shuffle(X_train, y_train)
            

'''
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(ch, row, col),
        output_shape=(ch, row, col)))
model.add(... finish defining the rest of your model architecture here ...)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= /
            len(train_samples), validation_data=validation_generator, /
            nb_val_samples=len(validation_samples), nb_epoch=3)
            
'''


# In[13]:


'''tg = train_samples[0:10]
mg = generator(tg, batch_size=10)
mg,vg = next(generator(tg, batch_size=10))
mg = np.array(mg)
vg = np.array(vg)
print(mg.shape)
print(vg.shape)'''


# In[14]:


'''plt.imshow(mg[2])
plt.show()
print(vg)'''


# In[15]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam


# In[16]:


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))

model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
#model.add(Dropout(0.5))
model.add(Dense(50))
#model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch= len(train_samples), 
                    validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model.h5')

