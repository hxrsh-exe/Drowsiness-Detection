import os # Operating system interfaces.
from keras.preprocessing import image # Module to preprocess images for deep learning models.
import matplotlib.pyplot as plt # Library for creating visualizations in Python.
import numpy as np 
from keras.utils import to_categorical # Utilities in Keras.
import random, shutil # Module to generate random numbers and high-level file operations.
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D
from keras.models import load_model

# Function to create a data generator for training and validation
def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=1, target_size=(24,24), class_mode='categorical'):
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale', class_mode=class_mode, target_size=target_size)

BS = 32 # Defining batch size
TS = (24,24)

# Ensure the paths are correct for macOS
train_path = 'data/train'
valid_path = 'data/valid'

train_batch = generator(train_path, shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator(valid_path, shuffle=True, batch_size=BS, target_size=TS)

SPE = len(train_batch.classes) // BS
VS = len(valid_batch.classes) // BS
print(SPE, VS)

# Building the model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(train_batch, validation_data=valid_batch, epochs=15, steps_per_epoch=SPE, validation_steps=VS)

# Save the model
model.save('models/cnnCat2.h5', overwrite=True)
