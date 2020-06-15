from tensorflow.keras.layers import (Conv3D, BatchNormalization, AveragePooling3D, concatenate, Lambda, SpatialDropout3D,
                          Activation, Input, GlobalAvgPool3D, Dense, Conv3DTranspose, add)
from tensorflow.keras.regularizers import l2 as l2_penalty
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from collections.abc import Sequence
import random
import os
import scipy
import collections
from itertools import repeat

import numpy as np
import pandas as pd
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

@tf.function
def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    F1 score: https://en.wikipedia.org/wiki/F1_score
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)


def invasion_acc(y_true, y_pred):
    binary_truth = y_true[:, -2] + y_true[:, -1]
    binary_pred = y_pred[:, -2] + y_pred[:, -1]
    return binary_accuracy(binary_truth, binary_pred)


def invasion_precision(y_true, y_pred):
    binary_truth = y_true[:, -2] + y_true[:, -1]
    binary_pred = y_pred[:, -2] + y_pred[:, -1]
    return precision(binary_truth, binary_pred)


def invasion_recall(y_true, y_pred):
    binary_truth = y_true[:, -2] + y_true[:, -1]
    binary_pred = y_pred[:, -2] + y_pred[:, -1]
    return recall(binary_truth, binary_pred)


def invasion_fmeasure(y_true, y_pred):
    binary_truth = y_true[:, -2] + y_true[:, -1]
    binary_pred = y_pred[:, -2] + y_pred[:, -1]
    return fmeasure(binary_truth, binary_pred)


def ia_acc(y_true, y_pred):
    binary_truth = y_true[:, -1]
    binary_pred = y_pred[:, -1]
    return binary_accuracy(binary_truth, binary_pred)


def ia_precision(y_true, y_pred):
    binary_truth = y_true[:, -1]
    binary_pred = y_pred[:, -1]
    return precision(binary_truth, binary_pred)


def ia_recall(y_true, y_pred):
    binary_truth = y_true[:, -1]
    binary_pred = y_pred[:, -1]
    return recall(binary_truth, binary_pred)


def ia_fmeasure(y_true, y_pred):
    binary_truth = y_true[:, -1]
    binary_pred = y_pred[:, -1]
    return fmeasure(binary_truth, binary_pred)
	
class DiceLoss:
    def __init__(self, beta=1., smooth=1.):
        self.__name__ = 'dice_loss_' + str(int(beta * 100))
        self.beta = float(beta)  # the more beta, the more recall
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        bb = self.beta * self.beta
        y_true_f = K.batch_flatten(K.cast(y_true,dtype='float32'))
        y_pred_f = K.batch_flatten(K.cast(y_pred,dtype='float32'))
        intersection = K.sum(y_true_f * y_pred_f, axis=-1)
        weighted_union = bb * K.sum(y_true_f, axis=-1) + \
                         K.sum(y_pred_f, axis=-1)
        score = -((1 + bb) * intersection + self.smooth) / \
                (weighted_union + self.smooth)
        return score
		
		
		
PARAMS = {
    'activation': lambda: Activation('relu'),  # the activation functions
    'bn_scale': True,  # whether to use the scale function in BN
    'weight_decay': 0.,  # l2 weight decay
    'kernel_initializer': 'he_uniform',  # initialization
    'first_scale': lambda x: x / 128. - 1.,  # the first pre-processing function
    'dhw': [40, 40, 40],  # the input shape
    'k': 16,  # the `growth rate` in DenseNet
    'bottleneck': 4,  # the `bottleneck` in DenseNet
    'compression': 2,  # the `compression` in DenseNet
    'first_layer': 32,  # the channel of the first layer
    'down_structure': [4, 4, 4],  # the down-sample structure
    'output_size': 1,  # the output number of the classification head
    'dropout_rate': None  # whether to use dropout, and how much to use
}


def _conv_block(x, filters):
    bn_scale = PARAMS['bn_scale']
    activation = PARAMS['activation']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    bottleneck = PARAMS['bottleneck']
    dropout_rate = PARAMS['dropout_rate']

    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    x = Conv3D(filters * bottleneck, kernel_size=(1, 1, 1), padding='same', use_bias=False,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2_penalty(weight_decay))(x)
    if dropout_rate is not None:
        x = SpatialDropout3D(dropout_rate)(x)
    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    x = Conv3D(filters, kernel_size=(3, 3, 3), padding='same', use_bias=True,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2_penalty(weight_decay))(x)
    return x


def _dense_block(x, n):
    k = PARAMS['k']

    for _ in range(n):
        conv = _conv_block(x, k)
        x = concatenate([conv, x], axis=-1)
    return x


def _transmit_block(x, is_last):
    bn_scale = PARAMS['bn_scale']
    activation = PARAMS['activation']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    compression = PARAMS['compression']

    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    if is_last:
        x = GlobalAvgPool3D()(x)
    else:
        *_, f = x.get_shape().as_list()
        x = Conv3D(f // compression, kernel_size=(1, 1, 1), padding='same', use_bias=True,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=l2_penalty(weight_decay))(x)
        x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    return x


def get_model(weights=None, verbose=True, **kwargs):
    for k, v in kwargs.items():
        assert k in PARAMS
        PARAMS[k] = v
    if verbose:
        print("Model hyper-parameters:", PARAMS)

    dhw = PARAMS['dhw']
    first_scale = PARAMS['first_scale']
    first_layer = PARAMS['first_layer']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    down_structure = PARAMS['down_structure']
    output_size = PARAMS['output_size']

    shape = dhw + [1]

    inputs = Input(shape=shape)

    if first_scale is not None:
        scaled = Lambda(first_scale)(inputs)
    else:
        scaled = inputs
    conv = Conv3D(first_layer, kernel_size=(3, 3, 3), padding='same', use_bias=True,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=l2_penalty(weight_decay))(scaled)

    downsample_times = len(down_structure)
    top_down = []
    for l, n in enumerate(down_structure):
        db = _dense_block(conv, n)
        top_down.append(db)
        conv = _transmit_block(db, l == downsample_times - 1)

    feat = top_down[-1]
    for top_feat in reversed(top_down[:-1]):
        *_, f = top_feat.get_shape().as_list()
        deconv = Conv3DTranspose(filters=f, kernel_size=2, strides=2, use_bias=True,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=l2_penalty(weight_decay))(feat)
        feat = add([top_feat, deconv])
    seg_head = Conv3D(1, kernel_size=(1, 1, 1), padding='same',
                      activation='sigmoid', use_bias=True,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=l2_penalty(weight_decay),
                      name='seg')(feat)

    if output_size == 1:
        last_activation = 'sigmoid'
    else:
        last_activation = 'softmax'

    clf_head = Dense(output_size, activation=last_activation,
                     kernel_regularizer=l2_penalty(weight_decay),
                     kernel_initializer=kernel_initializer,
                     name='clf')(conv)

    model = Model(inputs, [clf_head, seg_head])
    if verbose:
        model.summary()

    if weights is not None:
        model.load_weights(weights)
    return model


def get_compiled(loss={"clf": 'categorical_crossentropy',
                       "seg": DiceLoss()},
                 optimizer='adam',
                 metrics={'clf': ['accuracy', precision, recall],
                          'seg': [precision, recall]},
                 loss_weights={"clf": 1., "seg": .2}, weights=None, **kwargs):
    model = get_model(weights=weights, **kwargs)
    model.compile(loss=loss, optimizer=optimizer,
                  metrics=metrics, loss_weights=loss_weights)
    return model
	
def get_gpu_session(ratio=None, interactive=False):
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    if ratio is None:
        config.gpu_options.allow_growth = True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = ratio
    if interactive:
        sess = tf.compat.v1.InteractiveSession(config=config)
    else:
        sess = tf.compat.v1.Session(config=config)
    return sess


def set_gpu_usage(ratio=None):
    sess = get_gpu_session(ratio)
    tf.compat.v1.keras.backend.set_session(sess)


def resize(voxel, spacing, new_spacing=[1., 1., 1.]):
    '''Resize `voxel` from `spacing` to `new_spacing`.'''
    resize_factor = []
    for sp, nsp in zip(spacing, new_spacing):
        resize_factor.append(float(sp) / nsp)
    resized = scipy.ndimage.interpolation.zoom(voxel, resize_factor, mode='nearest')
    for i, (sp, shape, rshape) in enumerate(zip(spacing, voxel.shape, resized.shape)):
        new_spacing[i] = float(sp) * shape / rshape
    return resized, new_spacing


def rotation(array, angle):
    '''using Euler angles method.
    @author: renchao
    @params:
        angle: 0: no rotation, 1: rotate 90 deg, 2: rotate 180 deg, 3: rotate 270 deg
    '''
    #
    X = np.rot90(array, angle[0], axes=(0, 1))  # rotate in X-axis
    Y = np.rot90(X, angle[1], axes=(0, 2))  # rotate in Y'-axis
    Z = np.rot90(Y, angle[2], axes=(1, 2))  # rotate in Z"-axis
    return Z


def reflection(array, axis):
    '''
    @author: renchao
    @params:
        axis: -1: no flip, 0: Z-axis, 1: Y-axis, 2: X-axis
    '''
    if axis != -1:
        ref = np.flip(array, axis)
    else:
        ref = np.copy(array)
    return ref


def crop(array, zyx, dhw):
    z, y, x = zyx
    d, h, w = dhw
    cropped = array[z - d // 2:z + d // 2,
              y - h // 2:y + h // 2,
              x - w // 2:x + w // 2]
    return cropped


def random_center(shape, move):
    offset = np.random.randint(-move, move + 1, size=3)
    zyx = np.array(shape) // 2 + offset
    return zyx

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


#Data
TRAIN_INFO = '/kaggle/input/sjtu-ee228-2020/train_val.csv'
TRAIN_NODULE_PATH = '/kaggle/input/sjtu-ee228-2020/train_val'
TEST_INFO = '/kaggle/input/sjtu-ee228-2020/sampleSubmission.csv'
TEST_NODULE_PATH = '/kaggle/input/sjtu-ee228-2020/test'

class Transform:
    '''The online data augmentation, including:
    1) random move the center by `move`
    2) rotation 90 degrees increments
    3) reflection in any axis
    '''

    def __init__(self, size, move):
        self.size = _triple(size)
        self.move = move

    def __call__(self, arr, aux=None):
        """
        :param arr:
        :param aux: segment array.
        :return:
        """
        shape = arr.shape
        if self.move is not None:
            center = random_center(shape, self.move)
            arr_ret = crop(arr, center, self.size)
            angle = np.random.randint(4, size=3)
            arr_ret = rotation(arr_ret, angle=angle)
            axis = np.random.randint(4) - 1
            arr_ret = reflection(arr_ret, axis=axis)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = rotation(aux_ret, angle=angle)
                aux_ret = reflection(aux_ret, axis=axis)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret
        else:
            center = np.array(shape) // 2
            arr_ret = crop(arr, center, self.size)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret


def load_TrainDataset(crop_size, move=3):
    transform = Transform(crop_size, move)
    df = pd.read_csv(TRAIN_INFO)
    label = df.values[:, 1]
    size = len(df)
    train_x_vox = []
    train_x_seg = []
    train_y = []
    for index in range(size):
        name = df.loc[index, 'name']
        label_clu = label[index]
        with np.load(os.path.join(TRAIN_NODULE_PATH, '%s.npz' % name)) as npz:
            voxel, seg = transform(npz['voxel'], npz['seg'])
        train_x_vox.append(voxel)
        train_x_seg.append(seg)
        train_y.append(label_clu)
    train_y = to_categorical(train_y, num_classes=2)
    print("shape :")
    print((np.array(train_x_vox)).shape)
    print((np.array(train_x_seg)).shape)
    return [np.array(train_x_vox),np.array(train_x_seg)], {"clf": train_y, "seg": np.array(train_x_seg)}

def load_TestDataset(crop_size, move=3):
    transform = Transform(crop_size, move)
    df = pd.read_csv(TEST_INFO)
    label = df.values[:, 1]
    size = len(df)
    test_x_vox = []
    test_x_seg = []
    test_y = []
    for index in range(size):
        name = df.loc[index, 'name']
        label_clu = label[index]
        with np.load(os.path.join(TEST_NODULE_PATH, '%s.npz' % name)) as npz:
            voxel, seg = transform(npz['voxel'], npz['seg'])
        test_x_vox.append(voxel)
        test_x_seg.append(seg)
        test_y.append(label_clu)

    return [np.array(test_x_vox),np.array(test_x_seg)]

#train and predict

def main(batch_sizes, crop_size, random_move, learning_rate,
         segmentation_task_ratio, weight_decay, save_folder, epochs):
    '''

    :param batch_sizes: the number of examples of each class in a single batch
    :param crop_size: the input size
    :param random_move: the random move in data augmentation
    :param learning_rate: learning rate of the optimizer
    :param segmentation_task_ratio: the weight of segmentation loss in total loss
    :param weight_decay: l2 weight decay
    :param save_folder: where to save the snapshots, tensorflow logs, etc.
    :param epochs: how many epochs to run
    :return:
    '''
    batch_size = sum(batch_sizes)
    train_x,train_y = load_TrainDataset(crop_size,random_move)
    model = get_compiled(output_size=2,
                                    optimizer=Adam(lr=learning_rate),
                                    loss={"clf": 'categorical_crossentropy',
                                          "seg": DiceLoss()},
                                    metrics={'clf': ['accuracy', precision, recall, 
                                                     invasion_acc, 
                                                     invasion_precision,invasion_recall,
                                                     ia_acc, 
                                                     ia_precision, ia_recall],
                                             'seg': [precision, recall]},
                                    loss_weights={"clf": 1., "seg": segmentation_task_ratio},
                                    weight_decay=weight_decay)

    checkpointer = ModelCheckpoint(filepath='tmp/%s/weights.{epoch:02d}.h5' % save_folder, verbose=1,
                                   period=1, save_weights_only=True)
    best_keeper = ModelCheckpoint(filepath='tmp/%s/best.h5' % save_folder, verbose=1, save_weights_only=True,
                                  monitor='val_clf_acc', save_best_only=True, period=1, mode='max')
    csv_logger = CSVLogger('tmp/%s/training.csv' % save_folder)
    tensorboard = TensorBoard(log_dir='tmp/%s/logs/' % save_folder)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.334, patience=10,
                                   verbose=1, mode='min', epsilon=1.e-5, cooldown=2, min_lr=0)

    model.fit(train_x,train_y, batch_size=batch_size, 
                         epochs=epochs, validation_split=0.005,shuffle = True,
                        callbacks=[checkpointer,best_keeper, lr_reducer, csv_logger, tensorboard])
    return model
    
set_gpu_usage()
model =main(batch_sizes=[8, 8, 8, 8],
     crop_size=[40, 40, 40],
     random_move=3,
     learning_rate=1.e-4,
     segmentation_task_ratio=0.2,
     weight_decay=0.,
     save_folder='test',
     epochs=50)
test_x = load_TestDataset([40, 40, 40],3)

pre_y=model.predict(test_x)
#print(pre_y)

#to_csc
def load_result(pre_y):
    df = pd.read_csv(TEST_INFO)
    for index in range(len(df)):
        df.loc[index, 'predicted']=pre_y[index][1]
    df.to_csv("new_result_50.csv", index=False)
	
load_result(pre_y[0])