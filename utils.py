import numpy as np
import cv2 as cv
import pandas as pd
import os
from scipy import misc
from scipy import ndimage as ndi
from skimage.feature import canny

from parameters import *


def match_pred_yd(predictions):
    dataframe = pd.read_csv(CSV_PATH)
    l = []
    for i in range(len(predictions)):
        l.append(int(dataframe['ClassId'].loc[dataframe['ModelId'] == predictions[i]]))
    return np.array(l)


def match_pred_ym(predictions):
    dataframe = pd.read_csv(CSV_PATH)
    l = []
    for i in range(len(predictions)):
        l.append(int(dataframe['ModelId'].loc[dataframe['ClassId'] == predictions[i]]))
    return l


def save_in_distribution_attack(model, attack_type, is_target, class_id, x, result):
    csv_data_attack = pd.DataFrame(columns=['path_adversarial', 'original_prevision', 'adversarial_prevision', 'success'])
    size = len(x)

    for i in range(len(x)):

        original_image = x[i]
        img_orig = np.expand_dims(original_image, axis=0)
        res_orig = model.predict(img_orig)
        Ypred_orig = np.argmax(res_orig, axis=1)

        if (attack_type == "FG"):
            adv1 = result[i]
            img_adv1 = np.expand_dims(adv1, axis=0)
            res_adv1 = model.predict(img_adv1)
            Ypred_adv1 = np.argmax(res_adv1, axis=1)

            if (is_target):
                if (int(Ypred_adv1[0]) != int(class_id)):
                    csv_data_attack.loc[i] = [ADV_IN_FG_ATTACK + "target/" + str(i) + ".png", Ypred_orig[0],
                                              Ypred_adv1[0], 0]
                else:
                    csv_data_attack.loc[i] = [ADV_IN_FG_ATTACK + "target/" + str(i) + ".png", Ypred_orig[0],
                                              Ypred_adv1[0], 1]
                cv.imwrite(ADV_IN_FG_ATTACK + "target/" + str(i) + '.png', adv1 * 255)
            else:
                if (int(Ypred_adv1[0]) != int(Ypred_orig[0])):
                    csv_data_attack.loc[i] = [ADV_IN_FG_ATTACK + "untarget/" + str(i) + ".png", Ypred_orig[0],
                                              Ypred_adv1[0], 1]
                else:
                    csv_data_attack.loc[i] = [ADV_IN_FG_ATTACK + "untarget/" + str(i) + ".png", Ypred_orig[0],
                                              Ypred_adv1[0], 0]
                cv.imwrite(ADV_IN_FG_ATTACK + "untarget/" + str(i) + '.png', adv1 * 255)

        else:
            adv2 = result[i]
            img_adv2 = np.expand_dims(adv2, axis=0)
            res_adv2 = model.predict(img_adv2)
            Ypred_adv2 = np.argmax(res_adv2, axis=1)

            if (is_target):
                if (int(Ypred_adv2[0]) != int(class_id)):
                    csv_data_attack.loc[i] = [ADV_IN_IT_ATTACK + "target/" + str(i) + ".png", Ypred_orig[0],
                                              Ypred_adv2[0], 0]
                else:
                    csv_data_attack.loc[i] = [ADV_IN_IT_ATTACK + "target/" + str(i) + ".png", Ypred_orig[0],
                                              Ypred_adv2[0], 1]

                cv.imwrite(ADV_IN_IT_ATTACK + "target/" + str(i) + '.png', adv2 * 255)

            else:
                if (int(Ypred_adv2[0]) != int(Ypred_orig[0])):
                    csv_data_attack.loc[i] = [ADV_IN_IT_ATTACK + "untarget/" + str(i) + ".png", Ypred_orig[0],
                                              Ypred_adv2[0], 1]
                else:
                    csv_data_attack.loc[i] = [ADV_IN_IT_ATTACK + "untarget/" + str(i) + ".png", Ypred_orig[0],
                                              Ypred_adv2[0], 0]
                cv.imwrite(ADV_IN_IT_ATTACK + "untarget/" + str(i) + '.png', adv2 * 255)

    count = 0
    for index, row in csv_data_attack.iterrows():
        if (row['success'] != 1):
            count += 1

    accuracy = round((1 - (float(count) / float(size))) * 100, 2)

    print("Attacks successful: " + str(accuracy) + "% ")

    if (attack_type == "FG"):
        if (is_target):
            csv_data_attack.to_csv(ADV_IN_FG_ATTACK + "target/result.csv", sep=',', index=False)
        else:
            csv_data_attack.to_csv(ADV_IN_FG_ATTACK + "untarget/result.csv", sep=',', index=False)
    else:
        if (is_target):
            csv_data_attack.to_csv(ADV_IN_IT_ATTACK + "target/result.csv", sep=',', index=False)
        else:
            csv_data_attack.to_csv(ADV_IN_IT_ATTACK + "untarget/result.csv", sep=',', index=False)


def save_out_distribution_attack(model, attack_type, class_id, method, x, result):
    csv_data_attack = pd.DataFrame(columns=['path_adversarial', 'adversarial_prevision', 'success'])
    size = len(x)

    for i in range(len(x)):

        if (attack_type == "FG"):
            adv1 = result[i]
            img_adv1 = np.expand_dims(adv1, axis=0)
            res_adv1 = model.predict(img_adv1)
            Ypred_adv1 = np.argmax(res_adv1, axis=1)

            if (method == "LOGO"):
                if (int(Ypred_adv1[0]) != int(class_id)):
                    csv_data_attack.loc[i] = [ADV_OUT_LOGO_FG_ATTACK + str(i) + ".png", Ypred_adv1[0], 0]
                else:
                    csv_data_attack.loc[i] = [ADV_OUT_LOGO_FG_ATTACK + str(i) + ".png", Ypred_adv1[0], 1]
                cv.imwrite(ADV_OUT_LOGO_FG_ATTACK + str(i) + '.png', adv1 * 255)
            else:
                if (int(Ypred_adv1[0]) != int(class_id)):
                    csv_data_attack.loc[i] = [ADV_OUT_BLANKS_FG_ATTACK + str(i) + ".png", Ypred_adv1[0], 0]
                else:
                    csv_data_attack.loc[i] = [ADV_OUT_BLANKS_FG_ATTACK + str(i) + ".png", Ypred_adv1[0], 1]
                cv.imwrite(ADV_OUT_BLANKS_FG_ATTACK + str(i) + '.png', adv1 * 255)
        else:
            adv2 = result[i]
            img_adv2 = np.expand_dims(adv2, axis=0)
            res_adv2 = model.predict(img_adv2)
            Ypred_adv2 = np.argmax(res_adv2, axis=1)

            if (method == "LOGO"):
                if (int(Ypred_adv2[0]) != int(class_id)):
                    csv_data_attack.loc[i] = [ADV_OUT_LOGO_IT_ATTACK + str(i) + ".png", Ypred_adv2[0], 0]
                else:
                    csv_data_attack.loc[i] = [ADV_OUT_LOGO_IT_ATTACK + str(i) + ".png", Ypred_adv2[0], 1]
                cv.imwrite(ADV_OUT_LOGO_IT_ATTACK + str(i) + '.png', adv2 * 255)
            else:
                if (int(Ypred_adv2[0]) != int(class_id)):
                    csv_data_attack.loc[i] = [ADV_OUT_BLANKS_IT_ATTACK + str(i) + ".png", Ypred_adv2[0], 0]
                else:
                    csv_data_attack.loc[i] = [ADV_OUT_BLANKS_IT_ATTACK + str(i) + ".png", Ypred_adv2[0], 1]
                cv.imwrite(ADV_OUT_BLANKS_IT_ATTACK + str(i) + '.png', adv2 * 255)

    count = 0
    for index, row in csv_data_attack.iterrows():
        if (row['success'] != 1):
            count += 1

    accuracy = round((1 - (float(count) / float(size))) * 100, 2)

    print("Attacks successful: " + str(accuracy) + "% ")
    
    if (attack_type == "FG"):
        if (method == "LOGO"):
            csv_data_attack.to_csv(ADV_OUT_LOGO_FG_ATTACK + "result.csv", sep=',', index=False)
        else:
            csv_data_attack.to_csv(ADV_OUT_BLANKS_FG_ATTACK + "result.csv", sep=',', index=False)
    else:
        if (method == "LOGO"):
            csv_data_attack.to_csv(ADV_OUT_LOGO_IT_ATTACK + "result.csv", sep=',', index=False)
        else:
            csv_data_attack.to_csv(ADV_OUT_BLANKS_IT_ATTACK + "result.csv", sep=',', index=False)


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    if iteration == total:
        print()


def resize(image, size=IMAGE_SIZE, interp='bilinear'):
    img = misc.imresize(image, size, interp=interp)
    img = (img / 255.).astype(np.float32)
    return img


def resize_all(images, interp='bilinear'):
    if images[0].ndim == 3:
        shape = (len(images),) + IMAGE_SIZE + (N_CHANNEL,)
    elif images[0].ndim == 2:
        shape = (len(images),) + IMAGE_SIZE
    else:
        return
    imgs = np.zeros(shape)
    for i, image in enumerate(images):
        imgs[i] = resize(image, interp=interp)
    return imgs


def find_sign_area(image, sigma=1):
    edges = canny(image, sigma=sigma)
    fill = ndi.binary_fill_holes(edges)
    label_objects, _ = ndi.label(fill)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = np.zeros_like(sizes)
    sizes[0] = 0
    mask_sizes[np.argmax(sizes)] = 1.
    cleaned = mask_sizes[label_objects]

    return cleaned


def read_images(path, resize=False, interp='bilinear'):
    imgs = []
    valid_images = [".jpg", ".gif", ".png", ".tga", ".jpeg", ".ppm"]
    for f in sorted(os.listdir(path)):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        im = misc.imread(os.path.join(path, f), flatten=False, mode='RGB')
        if resize:
            im = misc.imresize(im, (32, 32), interp=interp)
        im = (im / 255.).astype(np.float32)
        imgs.append(im)
    return np.array(imgs)


def read_labels(path):
    with open(path) as f:
        content = f.readlines()
    content = [int(x.strip()) for x in content]
    return content


def load_out_samples(img_dir):
    images = read_images(img_dir, True)
    masks_full = []

    for i, image in enumerate(images):
        mask = find_sign_area(rgb2gray(image))
        masks_full.append(mask)

    masks = resize_all(masks_full, interp='nearest')
    x_ben = resize_all(images, interp='bilinear')

    return x_ben, masks


def load_samples(img_dir, label_path, tg):
    images = read_images(img_dir, True)
    masks_full = []

    labels = read_labels(label_path)
    result = match_pred_ym(labels)

    rm_indx = -1
    if (tg in result):
        print("Deleting target class from samples...")
        rm_indx = result.index(tg)
        images = np.delete(images, [rm_indx], axis=0)
        result = np.delete(result, [rm_indx], axis=0)

    for i, image in enumerate(images):
        mask = find_sign_area(rgb2gray(image))
        masks_full.append(mask)

    masks = resize_all(masks_full, interp='nearest')
    x_ben = resize_all(images, interp='bilinear')

    return x_ben, result, masks


def detect_phase(image_path):
    img = np.copy(image_path)
    # mg_ratio defines the padding, 0.8 is too large
    box = find_circles(img, mg_ratio=0.4, n_circles=1)
    for b in box:
        crop = crop_image(image_path, b)
        resized_im = resize_image(crop)
    return resized_im


def detect_real_time_phase(image_path):
    img = np.copy(image_path)
    # mg_ratio defines the padding, 0.8 is too large
    box = find_circles(img, mg_ratio=0.8, n_circles=1)
    for b in box:
        crop = crop_image(image_path, b)
        resized_im = resize_image(crop)
    return resized_im, box


def rename_signs(csv_path, res):
    dataframe = pd.read_csv(CSV_PATH)
    r = dataframe['SignName'].loc[dataframe['ModelId'] == res]
    r = r.to_string(index=False, header=False)
    return r.strip()


def resize_image(image, size=32, interp='bilinear'):
    img = cv.resize(image, (size, size), interpolation=cv.INTER_AREA)
    img = (img / 255.).astype(np.float32)
    return img


def crop_image(img, bbox):
    bb = np.array(bbox)
    bb = bb * (bb > 0)
    return img[bb[1]:bb[3], bb[0]:bb[2], :]


def rgb2gray(image):
    if image.ndim == 3:
        return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] +
                0.114 * image[:, :, 2])
    elif image.ndim == 4:
        return (0.299 * image[:, :, :, 0] + 0.587 * image[:, :, :, 1] +
                0.114 * image[:, :, :, 2])


def find_circles(img, mg_ratio, n_circles):
    targetImg = np.copy(img)
    # cv.imwrite("Detector_samples/detection_phase_steps/original1.png", targetImg)

    targetImg = cv.GaussianBlur(targetImg, (13, 13), 0)
    # cv.imwrite("Detector_samples/detection_phase_steps/gaussian_blur1.png", targetImg)

    grayImg = np.uint8(rgb2gray(targetImg))
    # cv.imwrite("Detector_samples/detection_phase_steps/gray_img1.png", grayImg)

    circles = cv.HoughCircles(grayImg, cv.HOUGH_GRADIENT, 1, 200, param1=50, param2=30, minRadius=20, maxRadius=250)
    boxes_list = []
    try:
        cir = circles.astype(np.uint16)
        for c in cir[0, :n_circles]:
            r = int(c[2])
            mg = int(r * mg_ratio)
            boxes_list.append([c[0] - r - mg, c[1] - r - mg, c[0] + r + mg, c[1] + r + mg])
    except AttributeError:
        pass
    except:
        raise
    return boxes_list
