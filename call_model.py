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
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=correct, logits=predicted)

def build_cnn(input_shape, num_classes):
    
    inpt = Input(shape=input_shape)
    conv_1 = Convolution2D(32, (5, 5), padding='same', activation='relu')(inpt)
    drop_1 = Dropout(rate=0.2)(conv_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(drop_1)

    conv_2 = Convolution2D(64, (5, 5), padding='same', activation='relu')(pool_1)
    drop_2 = Dropout(rate=0.3)(conv_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(drop_2)

    conv_3 = Convolution2D(128, (5, 5), padding='same', activation='relu')(pool_2)
    drop_3 = Dropout(rate=0.4)(conv_3)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(drop_3)

    concat = Concatenate(axis=-1)([Flatten()(pool_1), Flatten()(pool_2), Flatten()(pool_3)])
    dense_1 = Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001))(concat)
    drop_4 = Dropout(rate=0.5)(dense_1)
    output = Dense(num_classes, kernel_regularizer=keras.regularizers.l2(0.0001))(drop_4)

    model = keras.models.Model(inputs=inpt, outputs=output)

    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=loss_out, metrics=['accuracy'])

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

def plt_model():
	model = load_model()
	model.summary()

#plt_model()
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