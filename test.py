import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import warnings

from parameters import *
import call_model
import utils

warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    module=r'.*TiffImagePlugin'
)

model = call_model.load_model()

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DATA_DIR_2,
    target_size=(HEIGHT, WIDTH),
    batch_size=32,
    class_mode=None,
    shuffle=False)

val_steps = test_generator.n // test_generator.batch_size + 1

preds = model.predict_generator(test_generator, verbose=1, steps=val_steps)
Ypred = np.argmax(preds, axis=1)

test = pd.read_csv(CSV_PATH_TEST)
count_err = 0

# Solve labels problem
result = utils.match_pred_yd(Ypred)

for i, item in test.iterrows():
    if (item['ClassId'] != result[i]):
        count_err += 1

accuracy = round((1 - (float(count_err) / float(len(Ypred)))) * 100, 2)
print("Accuracy " + str(accuracy) + "%")
