import numpy as np
import cv2
import warnings
import numpy as np
from sklearn.utils import shuffle
from skimage import exposure
import random
from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform
from skimage.exposure import adjust_gamma
import os, shutil

from parameters import *


def apply_flip(X, y):
	flippable_horizontally = np.array([11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
	flippable_vertically = np.array([1, 5, 12, 15, 17])

	if(y in flippable_horizontally):
		X = np.fliplr(X)
	elif(y in flippable_vertically):
		X = np.flipud(X)
	else:
		X = X
	return X

def apply_brightness(X, intensity):
	delta = 1. * intensity
	X = adjust_gamma(X, random.uniform(1 - delta, 1 + delta))
	return X

def apply_rotate_image(X, intensity):
	delta = 30. * intensity
	X = rotate(X, random.uniform(-delta, delta))
	return X

def apply_projection_transform(X, intensity):
    image_size = X.shape[1]
    d = image_size * 0.3 * intensity

    tl_top = random.uniform(-d, d)
    tl_left = random.uniform(-d, d)
    bl_bottom = random.uniform(-d, d)
    bl_left = random.uniform(-d, d)
    tr_top = random.uniform(-d, d)
    tr_right = random.uniform(-d, d)
    br_bottom = random.uniform(-d, d)
    br_right = random.uniform(-d, d)

    transform = ProjectiveTransform()
    transform.estimate(np.array((
                (tl_left, tl_top),
                (bl_left, image_size - bl_bottom),
                (image_size - br_right, image_size - br_bottom),
                (image_size - tr_right, tr_top)
            )), np.array((
                (0, 0),
                (0, image_size),
                (image_size, image_size),
                (image_size, 0)
            )))

    X = warp(X, transform, output_shape=(image_size, image_size), order = 1, mode = 'edge')

    return X


def add_data(root, files, fixed_length):
	a = root.replace(TRAIN_PATH, '')
	os.makedirs(TRAIN_AUG_PATH+"/"+str(a[1:]))
	print("Adding samples to class "+ str(a[1:]))
	n = len(files)
	add_size = fixed_length-n
	l=[]
	for i in range(len(files)):
		if(files[i][:2]!="GT"):
			img = cv2.imread(root+"/"+files[i])
			l.append(img)

	l2=[]
	while(len(l2)<=add_size):
		rnd_img = l[random.randint(0,len(l)-1)]
		rnd_num = random.randint(1,4)
		img = None
		if(rnd_num == 1):
			img = apply_brightness(rnd_img, 0.5)
		elif(rnd_num == 2):
			img = apply_rotate_image(rnd_img, 0.5)
		elif(rnd_num == 3):
			img = apply_projection_transform(rnd_img, 0.5)
		else:
			img = apply_flip(rnd_img, int(a[1:]))

		l2.append(rnd_img)

	l_tot = l + l2
	for j in range(len(l_tot)):
		cv2.imwrite(TRAIN_AUG_PATH+"/"+str(a[1:])+"/"+str(j)+".png", l_tot[j])


def rem_data(root, files, fixed_length):
	a = root.replace(TRAIN_PATH, '')
	os.makedirs(TRAIN_AUG_PATH+"/"+str(a[1:]))
	print("Removing samples from class "+ str(a[1:]))
	n = len(files)
	rem_size = n-fixed_length
	l=[]
	for i in range(n):
		if(files[i][:2]!="GT"):
			img = cv2.imread(root+"/"+files[i])
			l.append(img)

	for i in range(len(l)):
		if(rem_size>1):
			if(i%3 == 0):
				del l[i]
				rem_size-=1

	for j in range(len(l)):
		cv2.imwrite(TRAIN_AUG_PATH+"/"+str(a[1:])+"/"+str(j)+".png", l[j])



im = cv2.imread("aug_examples/00001_00011_00027.png")
img = apply_flip(im, 1)
cv2.imwrite("aug_examples/1.png", img)

img = apply_brightness(im, 0.5)
cv2.imwrite("aug_examples/2.png", img)

img = apply_rotate_image(im, 0.5)
cv2.imwrite("aug_examples/3.png", img*255)

img = apply_projection_transform(im, 0.5)
cv2.imwrite("aug_examples/4.png", img*255)

exit()


print("Starting DATA AUGMENTATION\n")
if os.path.exists(TRAIN_AUG_PATH):
	shutil.rmtree(TRAIN_AUG_PATH)
	print("Removed existing dir...\n")

for root, dirs, files in os.walk(TRAIN_PATH):
	if(root != TRAIN_PATH):
		if (len(files) < CLASS_SIZE):
			add_data(root, files, CLASS_SIZE)
		elif (len(files) > CLASS_SIZE):
			rem_data(root, files, CLASS_SIZE)
		else:
			continue

print("Done!!!")