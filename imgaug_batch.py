from imgaug import augmenters as iaa
from PIL import Image
from numpy import *
import numpy as ny
import os
import os.path
from shutil import copyfile
from math import floor
from random import shuffle
import matplotlib.pylab as plt

seq = iaa.Sequential([
    # iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    # iaa.Fliplr(0.5), # horizontally flip 50% of the images
    # iaa.GaussianBlur(sigma=(0, 3.0)), # blur images with a sigma of 0 to 3.0
    # iaa.Flipud(0.5), #Flip 50% of all images vertically:
    # iaa.Crop(percent=(0, 0.1)),  # random crops
    # iaa.ContrastNormalization((0.75, 1.5)), # Strengthen or weaken the contrast in each image.
    # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # Add gaussian noise.
    # iaa.Multiply((0.8, 1.2), per_channel=0.2), # Make some images brighter and some darker.
    # iaa.Affine(translate_px={"x":-40}), # Augmenter to apply affine transformations to images.
    # iaa.AdditiveGaussianNoise(scale=0.1*255),
    # iaa.Scale({"height": 32, "width": 64})
    # iaa.Scale({"height": 128, "width": "keep-aspect-ratio"}),
    # iaa.Scale((0.5, 0.6)),
    # iaa.Scale({"height": (0.5, 0.75), "width": [64,128]}),
    # iaa.CropAndPad(percent=(-0.25, 0.25)),
    iaa.Sometimes(0.5,
        iaa.Scale({"height": 128, "width": "keep-aspect-ratio"}),
    ),
])

src_dir = './newFood_724_clean'

class_names = []
for filename in os.listdir(src_dir):
    path = os.path.join(src_dir, filename)
    if os.path.isdir(path):
        class_names.append(filename)

lst = class_names

imgs = []
for ind in range(0,len(lst)):
    sblst=os.listdir(os.path.join(src_dir,lst[ind]))
    # shuffle(sblst)
    print(len(sblst))
    cnt=0
    for pic_name in sblst:
        filepath_src= src_dir + '/' + lst[ind] + '/' + pic_name
        if pic_name.endswith('.db') == True:
            continue
        else:
            #transfer into numpy array here
            # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
            # or a list of 3D numpy arrays, each having shape (height, width, channels).
            # Grayscale images must have shape (height, width, 1) each.
            # All images must have numpy's dtype uint8. Values are expected to be in
            # range 0-255.
            cnt += 1
            img = plt.imread(filepath_src)
            imgs.append(img)

imgs = ny.array(imgs)
size = ny.size(imgs)
images_aug = seq.augment_images(imgs)
pil_im2 = Image.fromarray(uint8(images_aug))
pil_im2.show()