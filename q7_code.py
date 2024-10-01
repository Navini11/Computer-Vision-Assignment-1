import cv2
import matplotlib.pyplot as plt

# Load the Einstein image in grayscale
image_path = r'a1images\a1images\einstein.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Display the original image
plt.imshow(image, cmap='gray')
plt.title('Original Image - Einstein')
plt.show()

##########################################(a) Using cv2.filter2D to Sobel Filter the Image
#The cv2.filter2D function is used here to apply the Sobel filter.

import numpy as np

# Sobel kernels for X and Y directions
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

# Apply the Sobel filter using cv2.filter2D
sobel_filtered_x = cv2.filter2D(image, -1, sobel_x)
sobel_filtered_y = cv2.filter2D(image, -1, sobel_y)

# Display the Sobel filtered images
# Display the original image
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image - Einstein')

plt.subplot(1, 3, 2)
plt.imshow(sobel_filtered_x, cmap='gray')
plt.title('Sobel Filter-X Direction')

plt.subplot(1, 3, 3)
plt.imshow(sobel_filtered_y,cmap='gray')
plt.title('Sobel Filter-Y Direction')
plt.show()


###################################################Own Code for Sobel Filtering
#Here is the manual implementation of the Sobel filter using convolution.

def sobel_filter(image, kernel):
    [h, w] = np.shape(image) # Get rows and columns of the image
    output= np.zeros(shape=(h, w)) # Create empty image
    
    # Convolve the image with the kernel
    for i in range(1,h-1):
        for j in range(1, w-1):
            output[i, j] = np.sum(np.multiply(kernel,image[i-1:i + 2, j-1:j + 2]))
    return output

# Manually apply Sobel filters
sobel_manual_x = sobel_filter(image, sobel_x)
sobel_manual_y = sobel_filter(image, sobel_y)

# Display the manually filtered images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(sobel_manual_x,cmap='gray')
plt.title('Manual Sobel Filter - X Direction')

plt.subplot(1, 2, 2)
plt.imshow(sobel_manual_y, cmap='gray')
plt.title('Manual Sobel Filter - Y Direction')
plt.show()

############################################Using the Property
#We are going to use the separable Sobel filter property to convolve in two stages: first along the rows, then along the columns.

# Sobel filter property split into two 1D filters
kernel_x_row = np.array([[1, 0, -1]])  # 1x3
kernel_x_col = np.array([[1], [2], [1]])  # 3x1

# Apply separable Sobel filtering
sobel_separable_x = cv2.filter2D(image, -1, kernel_x_row)
sobel_separable_x = cv2.filter2D(sobel_separable_x, -1, kernel_x_col)

# Sobel filter in the Y direction using the same property
kernel_y_row = np.array([[1], [0], [-1]])  # 3x1
kernel_y_col = np.array([[1, 2, 1]])  # 1x3

sobel_separable_y = cv2.filter2D(image, -1, kernel_y_row)
sobel_separable_y = cv2.filter2D(sobel_separable_y, -1, kernel_y_col)

# Display the separable Sobel filtered images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(sobel_separable_x, cmap='gray')
plt.title('Separable Sobel Filter - X Direction')

plt.subplot(1, 2, 2)
plt.imshow(sobel_separable_y, cmap='gray')
plt.title('Separable Sobel Filter - Y Direction')
plt.show()








