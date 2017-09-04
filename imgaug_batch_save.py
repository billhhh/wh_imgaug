from imgaug import augmenters as iaa
from PIL import Image
from numpy import *
import numpy as np
import os
import os.path
from shutil import copyfile
from math import floor
from random import shuffle
import matplotlib.pylab as plt

seq = iaa.Sequential([
    iaa.Sometimes(0.5,
        iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
    ),
    iaa.Sometimes(0.5,
        iaa.Crop(percent=(0, 0.1)),  # random crops
    ),
    iaa.Flipud(0.5), #Flip 50% of all images vertically:
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0
    ),
    iaa.Sometimes(0.5,
        iaa.ContrastNormalization((0.75, 1.5)),  # Strengthen or weaken the contrast in each image.
    ),
    iaa.Sometimes(0.5,
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),  # Add gaussian noise.
    ),
    iaa.Sometimes(0.5,
        iaa.Multiply((0.8, 1.2), per_channel=0.2), # Make some images brighter and some darker.
    ),
    iaa.Sometimes(0.5,
        iaa.Affine(translate_px={"x":-40}), # Augmenter to apply affine transformations to images.
    ),
    iaa.AdditiveGaussianNoise(scale=0.1*255),
    iaa.Sometimes(0.4,
        iaa.Affine(translate_px={"x":-40}), # Augmenter to apply affine transformations to images.
    ),
    iaa.Sometimes(0.5,
        iaa.Scale({"height": 512, "width": 512})
    ),
], random_order=True) # apply augmenters in random order

src_dir = './newFood_724_clean'
class_names = []
for filename in os.listdir(src_dir):
    path = os.path.join(src_dir, filename)
    if os.path.isdir(path):
        class_names.append(filename)

lst = class_names
imgs = []
imgpath = []
for ind in range(0,len(lst)):
    sblst=os.listdir(os.path.join(src_dir,lst[ind]))
    # shuffle(sblst)
    print(len(sblst))
    for pic_name in sblst:
        filepath_src= src_dir + '/' + lst[ind] + '/' + pic_name
        imgpath.append(src_dir + '/' + lst[ind] + '/'+ 'aug'+pic_name)
        if pic_name.endswith('.db') == True:
            continue
        else:
            # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
            # or a list of 3D numpy arrays, each having shape (height, width, channels).
            # Grayscale images must have shape (height, width, 1) each.
            # All images must have numpy's dtype uint8. Values are expected to be in
            # range 0-255.
            img = plt.imread(filepath_src)
            imgs.append(img)

images_aug = seq.augment_images(imgs)

for i in range(0,len(imgs)):
    # plt.imshow(images_aug[i])
    plt.imsave(imgpath[i],images_aug[i])