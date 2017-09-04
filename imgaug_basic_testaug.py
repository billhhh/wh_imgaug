from imgaug import augmenters as iaa
from PIL import Image
from numpy import *

seq = iaa.Sequential([
    # iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    # iaa.Fliplr(0.5), # horizontally flip 50% of the images
    # iaa.GaussianBlur(sigma=(0, 3.0)), # blur images with a sigma of 0 to 3.0
    # iaa.Flipud(0.5), #Flip 50% of all images vertically:
    # iaa.Crop(percent=(0, 0.1)),  # random crops
    # iaa.ContrastNormalization((0.75, 1.5)), # Strengthen or weaken the contrast in each image.
    # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # iaa.Affine(translate_px={"x":-40}),
    # iaa.AdditiveGaussianNoise(scale=0.1*255),
    # iaa.Scale({"height": 32, "width": 64})
    # iaa.Scale({"height": 32, "width": "keep-aspect-ratio"})
    # iaa.Scale((0.5, 1.0))
    iaa.CropAndPad(percent=(-0.25, 0.25))
])

pil_img =Image.open('apple/apple_86.jpg')
pil_img.show()
img = array(pil_img)
images_aug = seq.augment_image(img)
pil_im2 = Image.fromarray(uint8(images_aug))
# pil_im2 = Image.fromarray(images_aug)
pil_im2.show()