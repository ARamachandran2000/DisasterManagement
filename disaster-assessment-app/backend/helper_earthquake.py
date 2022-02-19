from glob import glob
import os
import random
import scipy.misc
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from urllib.request import urlretrieve
from tqdm import tqdm
import os.path
import shutil
import zipfile
import cv2



# Assuming RGB Format

Label = namedtuple('Label', ['name', 'color'])

label_classes = [
    Label('unlabelled', (0, 0, 0)),
    Label('undamaged', (0, 255, 0)),
    Label('medium_damage', (0, 0, 255)),
    Label('high_damage', (255, 0, 0))

]


def load_data(image_paths):
    image_files = glob(image_paths +  '*.png')

    return image_files


def gen_test_output(sess, logits, keep_prob, image_pl, image_test, image_shape, label_colors):
    for f in image_test:
        image_file = f
        print("image_file= ",image_file)

        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        labels = sess.run([tf.argmax(tf.nn.softmax(logits), axis=-1)], {keep_prob: 1.0, image_pl: [image]})

        labels = labels[0].reshape(image_shape[0], image_shape[1])

        labels_colored = np.zeros((128,128,4))

        for lab in label_colors:
            label_mask = labels == lab
            labels_colored[label_mask] = np.array([*label_colors[lab],255])

        mask = scipy.misc.toimage(labels_colored, mode="RGBA")

        cv2.imwrite("check.png",labels_colored)
        init_img = scipy.misc.toimage(image)
        init_img.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(init_img)


def save_inference_samples(runs_dir, image_test, sess, image_shape, logits, keep_prob, input_image,
                           label_colors):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(sess, logits, keep_prob, input_image, image_test, image_shape,
                                    label_colors)
    for name, image in image_outputs:
        print(name)
        scipy.misc.imsave(os.path.join(output_dir, name), image)


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))
