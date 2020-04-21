import sys
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import keras
import time

import call_model
from parameters import *
import utils
from fg_attack import fg
from iterative_attack import iterative

data = pd.read_csv(CSV_PATH)
data.sort_values(by="ModelId", ascending=True, inplace=True)
print(" ID -> SIGN NAME\n")
for i, item in data.iterrows():
	print(" "+str(item['ModelId'])+" -> "+str(item['SignName']))
print("\n")

while True:
    try:
        target = int(input("Enter a class ID to attack (default 1): ") or "1")
        if(target<0 or target>42):
        	print("Insert a valid class ID")
        	continue
    except ValueError:
        print("Insert a number")
        continue
    else:
        break

dataframe = pd.read_csv(CSV_PATH)
target = int(target)

print("\nStart of the attack\n")

model = call_model.load_model()

print("++++++++++ IN DISTRIBUTION ATTACK ++++++++++\n")

x, y, masks = utils.load_samples(SAMPLE_IMG_DIR, SAMPLE_LABEL, target)

y_target = np.zeros((len(x))) + target
y_target = keras.utils.to_categorical(y_target, NUM_LABELS)

y_true = np.zeros((len(x))) + y
y_true = keras.utils.to_categorical(y_true, NUM_LABELS)

utils.printProgressBar(0, 100, prefix = 'Progress FAST GRADIENT TARGET ATTACK:', suffix = 'Complete', length = 50)
x_fg_target = fg(model, x, y_target, masks, True) # FG TARGET ATTACK
utils.printProgressBar(100, 100, prefix = 'Progress FAST GRADIENT TARGET ATTACK:', suffix = 'Complete', length = 50)
utils.save_in_distribution_attack(model, "FG", True, target, x, x_fg_target)

print("\n\n")

utils.printProgressBar(0, 100, prefix = 'Progress FAST GRADIENT UNTARGET ATTACK:', suffix = 'Complete', length = 50)
x_fg_untarget = fg(model, x, y_true, masks, False) # FG UNTARGET ATTACK
utils.printProgressBar(100, 100, prefix = 'Progress FAST GRADIENT UNTARGET ATTACK:', suffix = 'Complete', length = 50)
utils.save_in_distribution_attack(model, "FG", False, target, x, x_fg_untarget)

print("\n\n")

utils.printProgressBar(0, 100, prefix = 'Progress ITERATIVE TARGET ATTACK:', suffix = 'Complete', length = 50)
x_it_target = iterative(model, x, y_target, masks, True) # IT TARGET ATTACK
utils.printProgressBar(100, 100, prefix = 'Progress ITERATIVE TARGET ATTACK:', suffix = 'Complete', length = 50)
utils.save_in_distribution_attack(model, "IT", True, target, x, x_it_target)

print("\n\n")

utils.printProgressBar(0, 100, prefix = 'Progress ITERATIVE UNTARGET ATTACK:', suffix = 'Complete', length = 50)
x_it_untarget = iterative(model, x, y_true, masks, False) # IT UNTARGET ATTACK
utils.printProgressBar(100, 100, prefix = 'Progress ITERATIVE UNTARGET ATTACK:', suffix = 'Complete', length = 50)
utils.save_in_distribution_attack(model, "IT", False, target, x, x_it_untarget)

print("\n\n\n\n")

print("++++++++++ OUT OF DISTRIBUTION ATTACK ++++++++++\n")
print("+++++ LOGO ATTACK +++++\n")

x, masks = utils.load_out_samples(SAMPLES_IMG_DIR_LOGO)

y = np.zeros((len(x))) + target
y = keras.utils.to_categorical(y, NUM_LABELS)

utils.printProgressBar(0, 100, prefix = 'Progress FAST GRADIENT TARGET ATTACK:', suffix = 'Complete', length = 50)
x_fg_target = fg(model, x, y, masks, True) # IT TARGET ATTACK
utils.printProgressBar(100, 100, prefix = 'Progress FAST GRADIENT TARGET ATTACK:', suffix = 'Complete', length = 50)
utils.save_out_distribution_attack(model, "FG", target, "LOGO", x, x_fg_target)

print("\n\n")

utils.printProgressBar(0, 100, prefix = 'Progress ITERATIVE TARGET ATTACK:', suffix = 'Complete', length = 50)
x_it_target = iterative(model, x, y, masks, True) # IT TARGET ATTACK
utils.printProgressBar(100, 100, prefix = 'Progress ITERATIVE TARGET ATTACK:', suffix = 'Complete', length = 50)
utils.save_out_distribution_attack(model, "IT", target, "LOGO", x, x_it_target)

print("\n\n\n\n")

print("++++++++++ OUT OF DISTRIBUTION ATTACK ++++++++++\n")
print("+++++ BLANK SIGNS ATTACK +++++\n")

x, masks = utils.load_out_samples(SAMPLE_IMG_DIR_BLANK)

y = np.zeros((len(x))) + target
y = keras.utils.to_categorical(y, NUM_LABELS)

utils.printProgressBar(0, 100, prefix = 'Progress FAST GRADIENT TARGET ATTACK:', suffix = 'Complete', length = 50)
x_fg_target = fg(model, x, y, masks, True) # IT TARGET ATTACK
utils.printProgressBar(100, 100, prefix = 'Progress FAST GRADIENT TARGET ATTACK:', suffix = 'Complete', length = 50)
utils.save_out_distribution_attack(model, "FG", target, "BLANK", x, x_fg_target)

print("\n\n")

utils.printProgressBar(0, 100, prefix = 'Progress ITERATIVE TARGET ATTACK:', suffix = 'Complete', length = 50)
x_it_target = iterative(model, x, y, masks, True) # IT TARGET ATTACK
utils.printProgressBar(100, 100, prefix = 'Progress ITERATIVE TARGET ATTACK:', suffix = 'Complete', length = 50)
utils.save_out_distribution_attack(model, "IT", target, "BLANK", x, x_it_target)

print("\n\n")
