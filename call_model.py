import keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import warnings
from keras.layers import Input, Convolution2D, Dropout, MaxPooling2D, Flatten, Concatenate, Dense
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from parameters import *
import utils

def loss_out(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)

def build_cnn(input_shape, num_classes):
    
    inpt = Input(shape=input_shape)
    conv1 = Convolution2D(32, (5, 5), padding='same', activation='relu')(inpt)
    drop1 = Dropout(rate=0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)

    conv2 = Convolution2D(64, (5, 5), padding='same', activation='relu')(pool1)
    drop2 = Dropout(rate=0.3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = Convolution2D(128, (5, 5), padding='same', activation='relu')(pool2)
    drop3 = Dropout(rate=0.4)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    merge = Concatenate(axis=-1)([flat1, flat2, flat3])
    dense1 = Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001))(merge)
    drop4 = Dropout(rate=0.5)(dense1)
    output = Dense(num_classes, activation=None, kernel_regularizer=keras.regularizers.l2(0.0001))(drop4)
    model = keras.models.Model(inputs=inpt, outputs=output)

    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, epsilon=1e-08), loss=loss_out, metrics=['accuracy'])

    return model

def load_model_weights(problem):
    filename = os.path.join(problem)
    try:
        model=build_cnn((WIDTH, HEIGHT, 3), 43)
        model.load_weights(filename)
        print("\nModel weights successfully from file %s\n" %filename)
    except OSError:    
        print("\nModel file %s not found!!!\n" %filename)
        model = None
    return model

def load_model():
    model = load_model_weights(MODEL_PATH)
    return model

def get_prediction(image_path):
    model = load_model()
    WIDTH, HEIGHT = 32, 32
    img = image.load_img(image_path, target_size = (WIDTH, HEIGHT))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    res = model.predict(img)
    Ypred = np.argmax(res, axis=1)
    dataframe = pd.read_csv(CSV_PATH)
    r = dataframe['SignName'].loc[dataframe['ModelId'] == Ypred[0]]
    return r.to_string(index=False, header=False)

def get_simple_prediction(image_path):
    model = load_model()
    WIDTH, HEIGHT = 32, 32
    img = image.load_img(image_path, target_size = (WIDTH, HEIGHT))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    res = model.predict(img)
    Ypred = np.argmax(res, axis=1)
    return Ypred[0]

def get_real_time_prediction(img, model):
    img = np.expand_dims(img, axis=0)
    res = model.predict(img)
    Ypred = np.argmax(res, axis=1)
    dataframe = pd.read_csv(CSV_PATH)
    r = dataframe['SignName'].loc[dataframe['ModelId'] == Ypred[0]]
    return r.to_string(index=False, header=False)


#model = load_model_weights(MODEL_PATH)
#model.summary()
#exit()
'''
entries = os.listdir(SAMPLE_IMG_DIR)
for entry in entries:
    if(entry!="labels.txt"):
        res = get_prediction(SAMPLE_IMG_DIR+"/"+entry)
        print(entry)
        print(res)'''