import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image = cv2.imread('./logo.png', cv2.IMREAD_UNCHANGED)
print(image.shape)
cv2.imshow('image', image)

bgr_image = image[:, :, :3]
alpha = image[:, :, 3]

_, binary_image= cv2.threshold(bgr_image, thresh= 0, maxval = 255, type = cv2.THRESH_BINARY)
print(binary_image.shape)
cv2.imshow('binary', binary_image)

for r in range(binary_image.shape[0]):
    for c in range(binary_image.shape[1]):
        if binary_image[r, c, 0] == 255 and binary_image[r, c, 1] == 255 and binary_image[r, c, 2] == 255:
            binary_image[r, c, :] = np.array([80, 127, 255])
        elif binary_image[r, c, 0] == 0 and binary_image[r, c, 1] == 0 and binary_image[r, c, 2] == 0 and image[r, c, 3] != 0:
            binary_image[r, c, :] = np.array([80, 127, 255])
cv2.imshow('orange logo', binary_image)

image[:, :, :3] = binary_image
print(binary_image.shape)
cv2.imshow('transformed', image)

image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
image.save('./logo.png')
plt.imshow(image)
plt.show()

cv2.waitKey(0)
