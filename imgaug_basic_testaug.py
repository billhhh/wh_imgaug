from imgaug import augmenters as iaa
from PIL import Image
from numpy import *
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
    # iaa.Scale({"height": 32, "width": 64}),
    # iaa.Scale({"height": 128, "width": "keep-aspect-ratio"}),
    # iaa.Scale((0.5, 0.6)),
    # iaa.Scale({"height": (0.5, 0.75), "width": [64,128]}),
    # iaa.CropAndPad(percent=(-0.25, 0.25)),
    # iaa.WithChannels(0, iaa.Add((10, 100))),
    # iaa.Superpixels(p_replace=0.5, n_segments=64),
    # iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB"),
    # iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
    # iaa.Add((-40, 40))
    # iaa.Multiply((0.5, 1.5), per_channel=0.5)
    # iaa.CoarseDropout(0.02, size_percent=0.5)
    # iaa.ContrastNormalization((0.5, 1.5))
    iaa.Affine(rotate=(-45, 45))
])

a=0
for i in range(0,1):
    # pil_img =Image.open('newFood_724_clean/apple/apple_86.jpg')
    # pil_img.show()
    # img = array(pil_img)
    # plt.imshow('D:/dataset/newFood_724_clean/images/beef_ball_kway_teow_soup/beef_ball_kway_teow_soup115.jpg')
    try:
        a+=1
        img = plt.imread('newFood_724_clean/apple/apple_86.jpg')
        plt.imshow(img)
        images_aug = seq.augment_image(img)
        pil_im2 = Image.fromarray(uint8(images_aug))
        # pil_im2 = Image.fromarray(images_aug)
        pil_im2.show()

    except(IOError ,ZeroDivisionError),e:
        print e
        continue

print "done!"