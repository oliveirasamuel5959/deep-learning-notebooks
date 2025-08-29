# Required libraries
import os
import cv2
import matplotlib.pyplot as plt 

image_path = "images/image.jpg"
image = cv2.imread(image_path)

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh = 127
im_bw = cv2.threshold(grayscale_image, thresh, 255, cv2.THRESH_BINARY)[1]

# Define figure and axis to plot
fig, axs = plt.subplots(1, 3)

# Read color image
axs[0].imshow(image, cmap='gray')
axs[0].set_title("color image")

# Read grayscale image
axs[1].imshow(grayscale_image, cmap='gray')
axs[1].set_title("grayscale image")

# Read binary image
axs[2].imshow(im_bw, cmap='gray')
axs[2].set_title("binary image")

fig.tight_layout()
plt.show()