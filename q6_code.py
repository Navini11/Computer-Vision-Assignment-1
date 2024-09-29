##a) Split into Hue, Saturation, and Value Planes
import cv2
import matplotlib.pyplot as plt

# Load the image in color
image_path = r'a1images\a1images\jeniffer.jpg'
image = cv2.imread(image_path)

# Convert the image from BGR to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Split the HSV image into hue, saturation, and value channels
hue, saturation, value = cv2.split(hsv_image)

# Display the hue, saturation, and value planes in grayscale
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(hue, cmap='gray')
plt.title('Hue Plane')

plt.subplot(1, 3, 2)
plt.imshow(saturation, cmap='gray')
plt.title('Saturation Plane')

plt.subplot(1, 3, 3)
plt.imshow(value, cmap='gray')
plt.title('Value Plane')

plt.show()



## b) Select Plane and Threshold to Extract Foreground Mask
# Threshold the value plane to extract the foreground
_, mask = cv2.threshold(value, 120, 255, cv2.THRESH_BINARY)

# Display the mask (foreground extraction)
plt.imshow(mask, cmap='gray')
plt.title('Foreground Mask')
plt.show()



## c) Obtain Foreground Using cv2.bitwise_and and Compute Histogram
# Apply the mask to extract the foreground from the value plane
foreground = cv2.bitwise_and(value, value, mask=mask)

# Compute the histogram of the foreground
hist_foreground = cv2.calcHist([foreground], [0], mask, [256], [0, 256])

# Display the foreground image and its histogram
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(foreground, cmap='gray')
plt.title('Foreground (Value Plane)')

plt.subplot(1, 2, 2)
plt.plot(hist_foreground)
plt.title('Histogram of Foreground')
plt.show()



## (d) Obtain the Cumulative Sum of the Histogram
import numpy as np

# Calculate the cumulative sum of the foreground histogram
cdf_foreground = np.cumsum(hist_foreground)

# Normalize the cumulative sum
cdf_normalized = cdf_foreground * hist_foreground.max() / cdf_foreground.max()

# Plot the cumulative distribution function (CDF)
plt.plot(cdf_normalized, color='b')
plt.title('Cumulative Sum of Foreground Histogram')
plt.show()



## (e) Histogram Equalization of the Foreground Using Formulas
# Mask all zeros to avoid division by zero
cdf_m = np.ma.masked_equal(cdf_foreground, 0)

# Perform histogram equalization: scale the values and fill masked zeros
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')

# Apply the equalized CDF to the foreground image
equalized_foreground = cdf_final[foreground]

# Show the histogram-equalized foreground
plt.imshow(equalized_foreground, cmap='gray')
plt.title('Histogram Equalized Foreground')
plt.show()




##(f) Extract Background and Combine with Histogram Equalized Foreground
# Invert the mask to obtain the background
background_mask = cv2.bitwise_not(mask)

# Extract the background using the inverted mask
background = cv2.bitwise_and(value, value, mask=background_mask)

# Combine the background with the equalized foreground
final_image = cv2.add(background, equalized_foreground)

# Replace the value plane in the original HSV image with the final image
hsv_image[:, :, 2] = final_image

# Convert back to BGR for display
result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

# Show the original image and the result with histogram-equalized foreground
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title('Result with Histogram-Equalized Foreground')
plt.show()



