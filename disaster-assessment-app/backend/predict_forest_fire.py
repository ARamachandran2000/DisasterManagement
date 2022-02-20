from helper_forest_fire import *
import tensorflow as tf
# tf.disable_v2_behavior()
import time
import numpy as np
import cv2
import shutil
import os
import argparse

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

    # To find and print tensor names:
    # for op in graph.get_operations():
    # 	print(str(op.name))

    return input_image, keep_prob, layer3_out, layer4_out, layer7_out


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network. We already have the encoder part based on vgg.
    Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # 1x1 convolution for layer 7 from vgg
    layer7_conv_1x1 = tf.layers.conv2d(
        inputs=vgg_layer7_out,
        filters=num_classes,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    # First deconvolution layer with layer7 (after 1x1 convolution) as input
    deconv_layer1 = tf.layers.conv2d_transpose(
        layer7_conv_1x1,
        num_classes,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))  # Stride amount is cause upsampling by 2

    # 1x1 convolution for layer 4 from vgg
    layer4_conv_1x1 = tf.layers.conv2d(
        vgg_layer4_out,
        num_classes,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    # Adding deconvolved layer 1 and layer 4 for first skip connection.
    skip_connection1 = tf.add(layer4_conv_1x1, deconv_layer1)

    # 1x1 convolution for layer 3 for skip connection 2
    layer3_conv_1x1 = tf.layers.conv2d(
        vgg_layer3_out,
        num_classes,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    # Second deconvolution layer
    deconv_layer2 = tf.layers.conv2d_transpose(
        skip_connection1,
        num_classes,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    # Second skip connection made up of second deconvolution layer and 1x1 convolution of layer 3
    skip_connection2 = tf.add(deconv_layer2, layer3_conv_1x1)

    # Final deconvolution layer to reconstruct image
    deconv_output_layer = tf.layers.conv2d_transpose(
        skip_connection2,
        num_classes,
        kernel_size=16,
        strides=8,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    return deconv_output_layer
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
    print(nn_last_layer.shape)
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    print(logits.shape)
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

    vgg_path = 'data/vgg/'
    runs_dir = '.temp/runs/'

    label_colors = {i: np.array(l.color) for i, l in enumerate(label_classes)}

    # Training Paramaters
    img_shape = (128, 128)
    num_classes = 4
    learning_rate = 1e-4

    with tf.Session() as sess:

        ## Construct Network ##
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        deconv_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        # placeholder for labels, shape=(128, 256, 512, 30). X placeholder is input_image found above.
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        ## Optimize Network ##
        logits, train_op, cross_entropy_loss = optimize(deconv_output, correct_label, learning_rate, num_classes)

        # Initialize variables
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('saved_models/forest-fire/'))
        graph = tf.get_default_graph()

        ### Testing ###
        image_test = load_data(image_paths)
        save_inference_samples(runs_dir, image_test, sess, img_shape, logits, keep_prob, input_image,
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

        # cv2.imshow('final', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        filename = test_img_path.split('/')[-1]
        exact_name, extension = filename.split('.')
        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        cv2.imwrite('outputs/{}'.format(exact_name+'_mask.'+extension), img)

        image = cv2.imread(test_img_path)
        alpha = 0.6
        dst = cv2.addWeighted(image, alpha , img, 1-alpha, 0)
        # cv2.imshow('final', dst)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite('outputs/{}'.format(test_img_path.split('/')[-1]), dst)
    
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
