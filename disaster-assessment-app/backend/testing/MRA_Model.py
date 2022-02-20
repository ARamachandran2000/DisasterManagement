import pywt
import tensorflow as tf
#from tensorflow import keras
import numpy as np
import cv2
from tensorflow.keras import models,layers
from wavetf import WaveTFFactory


def create_activation():
    return layers.ReLU()
def sigmoid_activation():
    return layers.Softmax()



def create_model():

    input_1 = layers.Input(batch_input_shape=(4,128,128,3))
    enc_1 = layers.Conv2D(64,kernel_size=(3,3),padding="same",kernel_initializer = tf.random_normal_initializer(stddev=0.01))(input_1)
    enc_1 = create_activation()(enc_1)
    enc_1 = layers.Conv2D(64,kernel_size=(3,3),padding="same",kernel_initializer = tf.random_normal_initializer(stddev=0.01))(enc_1)
    enc_1 = create_activation()(enc_1)    #This will be used in skip connection 1

    maxpool_1 = layers.MaxPooling2D((2,2))(enc_1)


    ######################################
    '''
    Take the DL1 block here and concatenate with the downstream layer
    '''

    w = WaveTFFactory().build('db2', dim=2)

    dl1_input = w.call(tf.expand_dims(input_1[:,:,:,0],axis=-1)) #Calculate Wavelet Decomposition on only gray scale image

    dl1_enc = layers.Conv2D(64,kernel_size=(3,3),padding="same",kernel_initializer = tf.random_normal_initializer(stddev=0.01))(dl1_input)
    dl1_enc = create_activation()(dl1_enc)
    dl1_enc = layers.Conv2D(64,kernel_size=(3,3),padding="same",kernel_initializer = tf.random_normal_initializer(stddev=0.01))(dl1_enc)
    dl1_enc = create_activation()(dl1_enc)


    #Concatenate with main mode i.e output of maxpool_1

    concat_1 = tf.concat([maxpool_1,dl1_enc],axis=-1)

    enc_2 = layers.Conv2D(128,kernel_size=(3,3),padding="same",kernel_initializer = tf.random_normal_initializer(stddev=0.01))(concat_1)
    enc_2 = create_activation()(enc_2)
    enc_2 = layers.Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(enc_2)
    enc_2 = create_activation()(enc_2) #This will be used in skip connection 2

    maxpool_2 = layers.MaxPooling2D((2,2))(enc_2)

    ######################################
    '''
    Take the DL2 block here and concatenate with the downstream layer
    '''

    dl2_input = w.call(tf.expand_dims(dl1_input[:, :, :, 0], axis=-1))  # Calculate Wavelet Decomposition on only channel 0 of dl1_input
    dl2_enc = layers.Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(dl2_input)
    dl2_enc = create_activation()(dl2_enc)
    dl2_enc = layers.Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(dl2_enc)
    dl2_enc = create_activation()(dl2_enc)



    concat_2 = tf.concat([maxpool_2,dl2_enc],axis=-1)

    enc_3 = layers.Conv2D(256, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(concat_2)
    enc_3 = create_activation()(enc_3)
    enc_3 = layers.Conv2D(256, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(enc_3)
    enc_3 = create_activation()(enc_3)  # This will be used in skip connection 3
    maxpool_3 = layers.MaxPooling2D((2, 2))(enc_3)

    enc_4 = layers.Conv2D(512, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(maxpool_3)
    enc_4 = create_activation()(enc_4)
    enc_4 = layers.Conv2D(512, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(enc_4)
    enc_4 = create_activation()(enc_4)  # This will be used in skip connection 4
    maxpool_4 = layers.MaxPooling2D((2, 2))(enc_4)

    enc_5 = layers.Conv2D(1024, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(maxpool_4)
    enc_5 = create_activation()(enc_5)
    enc_5 = layers.Conv2D(1024, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(enc_5)
    enc_5 = create_activation()(enc_5)

    #####-> Downsampling Layer Ends, from now onwards it is upsampling side


    upsampling_1 = layers.UpSampling2D((2,2))(enc_5)
    dec_1 = layers.Conv2D(512,kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(upsampling_1)
    dec_1 = create_activation()(dec_1)

    concat_3 = tf.concat([enc_4,dec_1],axis = -1)
    dec_2 = layers.Conv2D(512, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(concat_3)
    dec_2 = create_activation()(dec_2)
    dec_2 = layers.Conv2D(512, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(dec_2)
    dec_2 = create_activation()(dec_2)

    upsampling_2 = layers.UpSampling2D((2,2))(dec_2)
    dec_3 = layers.Conv2D(256, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(upsampling_2)
    dec_3 = create_activation()(dec_3)

    concat_4 = tf.concat([enc_3,dec_3],axis=-1)
    dec_4 = layers.Conv2D(256, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(concat_4)
    dec_4 = create_activation()(dec_4)
    dec_4 = layers.Conv2D(256, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(dec_4)
    dec_4 = create_activation()(dec_4)

    upsampling_3 = layers.UpSampling2D((2, 2))(dec_4)
    dec_5 = layers.Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(upsampling_3)
    dec_5 = create_activation()(dec_5)

    concat_5 = tf.concat([enc_2,dec_5],axis=-1)
    dec_6 = layers.Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(concat_5)
    dec_6 = create_activation()(dec_6)
    dec_6 = layers.Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(dec_6)
    dec_6 = create_activation()(dec_6)

    upsampling_4 = layers.UpSampling2D((2, 2))(dec_6)
    dec_7 = layers.Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(upsampling_4)
    dec_7 = create_activation()(dec_7)

    concat_6 = tf.concat([enc_1, dec_7], axis=-1)
    dec_8 = layers.Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(concat_6)
    dec_8 = create_activation()(dec_8)
    dec_8 = layers.Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(dec_8)
    dec_8 = create_activation()(dec_8)

    dec_9 = layers.Conv2D(4, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(dec_8)
    dec_9 = create_activation()(dec_9)
    final_output = layers.Conv2D(4 , kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(dec_9)
    #final_output = sigmoid_activation()(final_output)


    #model = models.Model(input_1,final_output)

    return final_output







































