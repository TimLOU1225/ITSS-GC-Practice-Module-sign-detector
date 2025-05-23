from __future__ import division
import tensorflow as tf
from keras.layers import Reshape, MaxPooling2D, Flatten, RepeatVector, UpSampling2D, Concatenate, BatchNormalization, Activation
from keras.layers import Conv2D
from keras.layers import ConvLSTM2D
from config import *
from gaussian_prior import LearningPrior
from keras import backend as K
import numpy as np

# Gaussian priors initialization
def gaussian_priors_init(shape, dtype=None, name=None, **kwargs):
    # 确保 dtype 参数被正确处理
    dtype = tf.float32 if dtype is None else dtype
    
    # 使用 TensorFlow 操作生成均匀分布的随机数
    means = tf.random.uniform(shape=(shape[0] // 2,), minval=0.3, maxval=0.7, dtype=dtype)
    covars = tf.random.uniform(shape=(shape[0] // 2,), minval=0.05, maxval=0.3, dtype=dtype)
    
    # 使用 TensorFlow 操作合并均值和方差
    initial_value = tf.concat([means, covars], axis=0)
    
    # 直接返回 initial_value，无需创建 tf.Variable
    # tf.Variable 的创建将由 add_weight 方法自动处理
    return initial_value

def sam_vgg(data):
    # conv_1
    trainable = True
    conv_1_out = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=trainable)(data[0])
    conv_1_out = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=trainable)(conv_1_out)

    ds_conv_1_out = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv_1_out)

    # conv_2
    conv_2_out = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=trainable)(ds_conv_1_out)
    conv_2_out = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=trainable)(conv_2_out)

    ds_conv_2_out = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv_2_out)

    # conv_3
    conv_3_out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=trainable)(ds_conv_2_out)
    conv_3_out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=trainable)(conv_3_out)
    conv_3_out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=trainable)(conv_3_out)

    ds_conv_3_out = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same')(conv_3_out)

    # conv_4
    conv_4_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=trainable)(ds_conv_3_out)
    conv_4_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=trainable)(conv_4_out)
    conv_4_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=trainable)(conv_4_out)

    ds_conv_4_out = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', padding='same')(conv_4_out)

    # conv_5 #
    conv_5_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=trainable)(ds_conv_4_out)
    conv_5_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=trainable)(conv_5_out)
    conv_5_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=trainable)(conv_5_out)

    s_conv_5_out = Conv2D(64, (3, 3), padding='same', activation='relu', name='s_conv_5', trainable=True)(conv_5_out)
    s_saliency_conv_5 = Conv2D(1, (1, 1), activation='sigmoid', name='s_saliency_conv_5', trainable=True)(s_conv_5_out)

    # attention from conv_5 #
    attention_conv_5_out = Flatten()(conv_5_out)
    attention_conv_5_out = RepeatVector(nb_timestep)(attention_conv_5_out)
    attention_conv_5_out = Reshape((nb_timestep, 14, 14, 512))(attention_conv_5_out)
    attention_conv_5 = (ConvLSTM2D(filters=512, kernel_size=(3, 3),
                                padding='same', return_sequences=False, stateful=False,
                                name='conv5_lstm1', trainable=trainable))(attention_conv_5_out)
    priors1 = LearningPrior(nb_gaussian=nb_gaussian, init=gaussian_priors_init)(data[1])
    attention_conv_5 = Concatenate()([attention_conv_5, priors1])
    attention_conv_5 = Conv2D(64, (3, 3), padding='same', activation='relu', name='merge_att_conv5',
                              trainable=trainable)(attention_conv_5)
    attention_conv_5 = Conv2D(1, (1, 1), activation='sigmoid', name='att_conv5', trainable=trainable)(attention_conv_5)

    conv_5_out = Concatenate()([s_conv_5_out, attention_conv_5])
    conv_5_out = Flatten()(conv_5_out)
    conv_5_out = RepeatVector(nb_timestep)(conv_5_out)
    conv_5_out = Reshape((nb_timestep, 14, 14, 65))(conv_5_out)
    saliency_conv_5 = (ConvLSTM2D(filters=64, kernel_size=(3, 3),
                                  padding='same', return_sequences=False, stateful=False,
                                  name='conv5_lstm2', trainable=True))(conv_5_out)
    saliency_conv_5 = Conv2D(1, (1, 1), activation='sigmoid', name='sal_conv5', trainable=True)(saliency_conv_5)

    conv_4_out = Conv2D(64, (1, 1), padding='same', name='conv_4_out', trainable=trainable)(conv_4_out)
    conv_4_out = BatchNormalization()(conv_4_out)
    conv_4_out = Activation('sigmoid')(conv_4_out)
    up_saliency_conv_5 = UpSampling2D(size=(2, 2))(saliency_conv_5)
    conv_4_out = Concatenate()([conv_4_out, up_saliency_conv_5])
    conv_4_out = Flatten()(conv_4_out)
    conv_4_out = RepeatVector(nb_timestep)(conv_4_out)
    conv_4_out = Reshape((nb_timestep, 28, 28, 65))(conv_4_out)

    saliency_conv_4 = (ConvLSTM2D(filters=64, kernel_size=(3, 3),
                                  padding='same', return_sequences=False, stateful=False,#True
                                  name='conv4_lstm2', trainable=True))(conv_4_out)
    saliency_conv_4 = Conv2D(1, (1, 1), activation='sigmoid', name='sal_conv4', trainable=True)(saliency_conv_4)

    # saliency from conv_3 #
    conv_3_out = Conv2D(64, (1, 1), padding='same', name='conv_3_out', trainable=True)(conv_3_out)
    conv_3_out = BatchNormalization()(conv_3_out)
    conv_3_out = Activation('sigmoid')(conv_3_out)
    up_saliency_conv_4 = UpSampling2D(size=(2, 2))(saliency_conv_4)
    conv_3_out = Concatenate()([conv_3_out, up_saliency_conv_4])
    conv_3_out = Flatten()(conv_3_out)
    conv_3_out = RepeatVector(nb_timestep)(conv_3_out)
    conv_3_out = Reshape((nb_timestep, 56, 56, 65))(conv_3_out)
    saliency_conv_3 = (ConvLSTM2D(filters=64, kernel_size=(3, 3),
                                   padding='same', return_sequences=False, stateful=False,
                                   name='conv3_lstm', trainable=True))(conv_3_out)
    saliency_conv_3 = Conv2D(1, (1, 1), activation='sigmoid', name='sal_conv3', trainable=True)(saliency_conv_3)

    # saliency from conv_2 #
    conv_2_out = Conv2D(64, (1, 1), padding='same', name='conv_2_out', trainable=True)(conv_2_out)
    conv_2_out = BatchNormalization()(conv_2_out)
    conv_2_out = Activation('sigmoid')(conv_2_out)
    up_saliency_conv_3 = UpSampling2D(size=(2, 2))(saliency_conv_3)
    conv_2_out = Concatenate()([conv_2_out, up_saliency_conv_3])
    conv_2_out = Flatten()(conv_2_out)
    conv_2_out = RepeatVector(nb_timestep)(conv_2_out)
    conv_2_out = Reshape((nb_timestep, 112, 112, 65))(conv_2_out)
    saliency_conv_2 = (ConvLSTM2D(filters=64, kernel_size=(3, 3),
                                   padding='same', return_sequences=False, stateful=False,
                                   name='conv2_lstm', trainable=True))(conv_2_out)
    saliency_conv_2 = Conv2D(1, (1, 1), activation='sigmoid', name='sal_conv2', trainable=True)(saliency_conv_2)

    # saliency from conv_1 #
    conv_1_out = Conv2D(32, (1, 1), padding='same', name='conv_1_out', trainable=True)(conv_1_out)
    conv_1_out = BatchNormalization()(conv_1_out)
    conv_1_out = Activation('sigmoid')(conv_1_out)
    up_saliency_conv_2 = UpSampling2D(size=(2, 2))(saliency_conv_2)
    conv_1_out = Concatenate()([conv_1_out, up_saliency_conv_2])
    conv_1_out = Flatten()(conv_1_out)
    conv_1_out = RepeatVector(nb_timestep)(conv_1_out)
    conv_1_out = Reshape((nb_timestep, 224, 224, 33))(conv_1_out)
    saliency_conv_1 = (ConvLSTM2D(filters=32, kernel_size=(3, 3),
                                   padding='same', return_sequences=False, stateful=False,
                                   name='conv1_lstm', trainable=True))(conv_1_out)
    saliency_conv_1 = Conv2D(1, (1, 1), activation='sigmoid', name='sal_conv1', trainable=True)(saliency_conv_1)

    return [attention_conv_5, s_saliency_conv_5, saliency_conv_5, saliency_conv_4,
            saliency_conv_3, saliency_conv_2, saliency_conv_1]