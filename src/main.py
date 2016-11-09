import numpy
import pylab
from PIL import Image
from deep_sergal import DeepSergal

img = Image.open(open('../pics/srgl-tel.jpg'))
img = numpy.asarray(img, dtype='float64') / 256

h, w, num_channels = img.shape

img_ = img.transpose(2, 0, 1).reshape(1, num_channels, h, w)
proccesed_img = DeepSergal().CNNTest(img_)

# plot original image and first and second components of output
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(proccesed_img[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(proccesed_img[0, 1, :, :])
pylab.show()