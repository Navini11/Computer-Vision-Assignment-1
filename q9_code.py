import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image_path = r'a1images\a1images\daisy.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initial mask (all zeros)
mask = np.zeros(image.shape[:2], np.uint8)

# Background and foreground models (used internally by GrabCut)
bg_model = np.zeros((1, 65), np.float64)
fg_model = np.zeros((1, 65), np.float64)

# Define a bounding box around the object (flower)
rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)

# Apply GrabCut algorithm
cv2.grabCut(image, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)

# Modify mask to get the foreground
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
foreground = image_rgb * mask2[:, :, np.newaxis]

# Extract the background
background = image_rgb * (1 - mask2[:, :, np.newaxis])

# Display segmentation results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(mask2, cmap='gray')
plt.title('Segmentation Mask')

plt.subplot(1, 3, 2)
plt.imshow(foreground)
plt.title('Foreground')

plt.subplot(1, 3, 3)
plt.imshow(background)
plt.title('Background')

plt.show()


# Apply Gaussian blur to the background
blurred_background = cv2.GaussianBlur(image_rgb, (35, 35), 0)

# Combine the blurred background with the foreground
enhanced_image = blurred_background * (1 - mask2[:, :, np.newaxis]) + foreground

# Display original and enhanced images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(enhanced_image.astype('uint8'))
plt.title('Enhanced Image with Blurred Background')

plt.show()
