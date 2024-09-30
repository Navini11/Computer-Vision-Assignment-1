import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'a1images/a1images/highlights_and_shadows.jpg'
image = cv.imread(image_path)

# Convert the image to the L*a*b* color space
lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)

# Split into L*, a*, b* channels
L, a, b = cv.split(lab_image)

#normalize L
L = L/255.0

# Apply gamma correction to the L* plane
gamma = 2.2  # Example gamma value
L_corrected = np.array( 255*(L) ** (gamma), dtype='uint8')
# 255 * (L^gamma)

# Merge the corrected L* channel back with the original a* and b* channels
lab_corrected = cv.merge((L_corrected, a, b))

# Convert back to BGR color space for display
image_corrected = cv.cvtColor(lab_corrected, cv.COLOR_LAB2BGR)

# Plot original and gamma-corrected images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(image_corrected, cv.COLOR_BGR2RGB))
plt.title('Gamma Corrected Image (γ = 2.2)')
plt.show()

# Plot histograms of the original and corrected L* planes
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(L.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.6, label='Original L*')
plt.title('Histogram of Original L*')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(L_corrected.ravel(), bins=256, range=(0, 256), color='red', alpha=0.6, label='Gamma Corrected L*')
plt.title('Histogram of Gamma Corrected L* (γ = 2.2)')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

