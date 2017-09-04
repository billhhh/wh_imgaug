from imgaug import augmenters as iaa
from PIL import Image
from numpy import *

seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])

pil_img =Image.open('apple/apple_86.jpg')
pil_img.show()
img = array(pil_img)
images_aug = seq.augment_image(img)
pil_im2 = Image.fromarray(uint8(images_aug))
pil_im2.show()