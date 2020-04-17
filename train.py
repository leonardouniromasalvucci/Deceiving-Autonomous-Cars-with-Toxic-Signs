import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import warnings
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

from parameters import *
import call_model


warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    module=r'.*TiffImagePlugin'
)

train_datagen = ImageDataGenerator(rescale=1./255,
    validation_split=0.3)

train_generator = train_datagen.flow_from_directory(
    TRAIN_AUG_PATH,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_AUG_PATH,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42)


tmp = pd.DataFrame(columns=['ClassId', 'ModelId', 'SignName'])
csv_data = pd.read_csv(CSV_PATH)
for i, item in csv_data.iterrows():
    tmp.loc[i] = [item['ClassId'], train_generator.class_indices[str(item['ClassId'])], item['SignName']]
tmp.to_csv(CSV_PATH, sep=',', index = False)

model = call_model.build_cnn((WIDTH, HEIGHT, 3), 43)

steps_per_epoch=train_generator.n//train_generator.batch_size
val_steps=validation_generator.n//validation_generator.batch_size+1

modelCheckpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=6, verbose=0, mode='auto')

callbacks_list = [modelCheckpoint, earlyStop]

history = model.fit_generator(
    train_generator,
    workers=6,
    epochs=EPOCHS,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    validation_data=validation_generator,
    callbacks=callbacks_list,
    shuffle=True)