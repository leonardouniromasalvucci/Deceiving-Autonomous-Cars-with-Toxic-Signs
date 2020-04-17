import numpy as np
import tensorflow as tf
import keras.backend as K

from parameters import *
import utils

def gradient_fn(model):
    y_true = K.placeholder(shape=(OUTPUT_DIM, ))
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=model.output)
    grad = K.gradients(loss, model.input)

    return K.function([model.input, y_true, K.learning_phase()], grad)

def gradient_input(grad_fn, x, y):
    
    return grad_fn([x.reshape(INPUT_SHAPE), y, 0])[0][0]

def iterative(model, x, y, mask, target):
    x_adv = np.zeros(x.shape, dtype=np.float32)
    grad_fn = gradient_fn(model)

    for i, x_in in enumerate(x):

        utils.printProgressBar(i+50, 100, prefix = 'Progress ITERATIVE TARGET ATTACK:', suffix = 'Complete', length = 50)

        x_cur = np.copy(x_in)
        mask_rep = np.repeat(mask[i, :, :, np.newaxis], N_CHANNEL, axis=2)

        for _ in range(60):
            
            if target == True:
                grad = -1 * gradient_input(grad_fn, x_cur, y[i])
            else:
                grad = gradient_input(grad_fn, x_cur, y[i])

            try:
                grad /= np.linalg.norm(grad)
            except ZeroDivisionError:
                raise

            grad *= mask_rep

            x_cur += grad * 0.09
            x_cur = np.clip(x_cur, 0, 1)
        
        x_adv[i] = np.copy(x_cur)

    return x_adv