import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

im = Image.open('bird.jpg')

#convert an image to grey scale using ITU-R 601-2 transform
image_gr = im.convert("L")

print("\n Original type: %r \n\n" %image_gr)

#convert image to a matrix with values from 0 to 255
arr = np.asarray(image_gr)

#plot image
imgplot = plt.imshow(arr)
imgplot.set_cmap('gray')

print("After conversion to gray: \n\n %r" %arr)
plt.show(imgplot)

#edge detector kernel
kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
grad = signal.convolve2d(arr, kernel, mode='same', boundary='symm')

print("GRADIENT MAGNITUDE - Feature map")

fig, aux = plt.subplots(figsize=(10,10))
aux.imshow(np.absolute(grad), cmap='gray')
print("After convolution")

plt.show(aux)
#normalize
grad_biases = np.absolute(grad) + 100
grad_biases[grad_biases > 255] = 255

print("GRADIENT MAGNITUDE - Feature map")
print('After normalization')
fig, aux = plt.subplots(figsize=(10,10))
aux.imshow(np.absolute(grad_biases), cmap='gray')
plt.show()
