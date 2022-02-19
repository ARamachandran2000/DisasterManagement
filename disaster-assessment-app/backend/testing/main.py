from helper import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import numpy as np
import cv2
import shutil
import os
import argparse


from MRA_Model import *

## Global variables ##
KEEP_PROB = 0.8
LEARNING_RATE = 1e-4

def load_vgg(sess, vgg_path):
    # Load vgg from path
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # Extract graph and tensors that we need to manipulate for deconvolutions
    # We also want input, keepprob, layer3, 4 and 7 outputs for this particular FCN.
    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    layer3_out = graph.get_tensor_by_name('layer3_out:0')
    layer4_out = graph.get_tensor_by_name('layer4_out:0')
    layer7_out = graph.get_tensor_by_name('layer7_out:0')

    return input_image, keep_prob, layer3_out, layer4_out, layer7_out


def create_activation():
    return layers.ReLU()
def sigmoid_activation():
    return layers.Softmax()



def create_model():

    input_1 = tf.placeholder(tf.float32, [1, 128, 128, 3], name='input')
    
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

    return input_1,final_output

# #############################################################################


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # logits and labels are now 2D tensors where each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # Computes softmax cross entropy between logits and labels
    cross_entropy_logits = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)

    # Computes the mean of elements across dimensions of a tensor
    cross_entropy_loss = tf.reduce_mean(cross_entropy_logits)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Minimizes loss by combining calls compute_gradients() and apply_gradients().
    train_op = optimizer.minimize(cross_entropy_loss, name='train_op')

    return logits, train_op, cross_entropy_loss

def pred(image_path):
    
    if os.path.exists('.temp'):
        shutil.rmtree('.temp')

    os.mkdir('.temp')
    os.mkdir('.temp/img_patches/')
    os.mkdir('.temp/runs/')

    test_img_path = image_path

    count = 0
    image = cv2.imread(test_img_path)
    for col in range(0, image.shape[0], 128):
        for row in range(0, image.shape[0], 128):
            patch = image[col:col+128, row:row+128, :]
            cv2.imwrite('.temp/img_patches/{}.png'.format(count), patch)
            count += 1

    # Load Data
    data_folder = '.temp/img_patches/'
    image_paths = data_folder

    vgg_path = '../data/vgg/'
    runs_dir = '.temp/runs/'

    label_colors = {i: np.array(l.color) for i, l in enumerate(label_classes)}

    # Training Paramaters
    img_shape = (128, 128)
    num_classes = 4
    learning_rate = 1e-4

    with tf.Session() as sess:

        ## Construct Network ##
        _, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        #deconv_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        input_1,deconv_output = create_model()

        # placeholder for labels, shape=(128, 256, 512, 30). X placeholder is input_image found above.
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        ## Optimize Network ##
        logits, train_op, cross_entropy_loss = optimize(deconv_output, correct_label, learning_rate, num_classes)

        # Initialize variables
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('../saved_models/'))
        graph = tf.get_default_graph()

        ### Testing ###
        image_test = load_data(image_paths)
        save_inference_samples(runs_dir, image_test, sess, img_shape, logits, keep_prob, input_1,
                               label_colors)

        # Load sample image and create dummy image
        sample_image_dir = 'sample_img.png'
        image = cv2.imread(sample_image_dir)
        img = np.zeros_like(image)

        # Stitch back patches
        row = 0
        col = -1

        for i in range(0, 64):
            if row % 8 == 0:
                col += 1

            _dir = os.listdir('.temp/runs/')
            patch = cv2.imread('.temp/runs/{}/{}.png'.format(_dir[0], i))
            img[col*128:(col+1)*128, (row%8)*128:((row%8)+1)*128, :] = patch

            row+=1

        cv2.imshow('final', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        cv2.imwrite('outputs/{}'.format(test_img_path.split('/')[-1]), img)
    
    if os.path.exists('.temp'):
        shutil.rmtree('.temp')

if __name__ == '__main__':
    # Creating parser
    my_parser = argparse.ArgumentParser(description='List the content of a folder')

    my_parser.add_argument('Path',
                       metavar='path',
                       type=str,
                       help='the path to list')

    args = my_parser.parse_args()
    image_path = args.Path

    # Predict
    pred(image_path)
