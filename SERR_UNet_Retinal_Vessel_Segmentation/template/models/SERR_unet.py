#!/usr/bin/env python
# -*-coding:utf-8 -*-
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from loss import dice_coef_loss,dice_coef,mean_iou
import keras


class SeBlock(keras.layers.Layer):
    def __init__(self, reduction=4, **kwargs):
        super(SeBlock, self).__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):  # 构建layer时需要实现
        # input_shape
        pass

    def call(self, inputs):
        x = keras.layers.GlobalAveragePooling2D()(inputs)
        x = keras.layers.Dense(int(x.shape[-1]) // self.reduction, use_bias=False, activation=keras.activations.relu)(x)
        x = keras.layers.Dense(int(inputs.shape[-1]), use_bias=False, activation=keras.activations.hard_sigmoid)(x)
        return keras.layers.Multiply()([inputs, x])  # 给通道加权重
        # return inputs*x

def rec_res_block(input_layer, out_n_filters, batch_normalization=True, kernel_size=[3, 3], stride=[1, 1],
                  padding='same', data_format='channels_last'):
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]
    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
        #skip_layer = Dropout(0.2)(skip_layer)
    else:
        skip_layer = input_layer
    layer = skip_layer
    for j in range(2):
        for i in range(2):
            if i == 0:
                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                    layer)
                #layer1 = Dropout(0.2)(layer1)
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = Activation('relu')(layer1)
            layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                add([layer1, layer]))
            #layer1 = Dropout(0.2)(layer1)
            if batch_normalization:
                layer1 = BatchNormalization()(layer1)
            layer1 = Activation('relu')(layer1)
        layer = layer1
    out_layer = add([layer, skip_layer])
    return out_layer

def up_and_concate(down_layer, layer, data_format='channels_last'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]
    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)
    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    concate = my_concat([up, layer])
    return concate

def get_SERR_unet(patch_height,patch_width,n_ch):
    inputs = Input(shape=(patch_height,patch_width,n_ch))
    x = inputs
    depth = 2
    features = 16
    skips = []
    for i in range(depth):
        x = rec_res_block(x,features,data_format='channels_last')
        skips.append(x)
        x = MaxPooling2D((2,2),data_format='channels_last')(x)
        features = features*2
    x = rec_res_block(x,features,data_format='channels_last')
    for i in reversed(range(depth)):
        features = features//2
        x = up_and_concate(x,skips[i],data_format='channels_last')
        x = rec_res_block(x,features,data_format='channels_last')
    outputs = SeBlock()(inputs)
    #conv6 = Conv2D(2,(1,1),activation='relu',padding='same',data_format='channels_last')(x)
    #conv6 = Reshape((2, patch_height * patch_width))(conv6)  # 此时output的shape是(batchsize,2,patch_height*patch_width)
    #conv6 = Permute((2, 1))(conv6)  # 此时output的shape是(Npatch,patch_height*patch_width,2)即输出维度是(Npatch,2304,2)
    conv6 = Conv2D(1, 1, activation='sigmoid')(x)
    ############
    model = Model(inputs=inputs, outputs=conv6)
    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[dice_coef,mean_iou])
    return model
if __name__ == '__main__':
    model = get_SERR_unet(400,400,1)
    model.summary()