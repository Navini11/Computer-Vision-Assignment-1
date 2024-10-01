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

# Create subplots for the original and gamma-corrected histograms
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Original image histograms (Red, Green, Blue in the same plot)
axs[0].set_title('Original Image - RGB Channels')
axs[0].hist(image[:, :, 0].flatten(), bins=256, range=[0, 256], color='b', alpha=0.4, label="Blue Channel")
axs[0].hist(image[:, :, 1].flatten(), bins=256, range=[0, 256], color='g', alpha=0.4, label="Green Channel")
axs[0].hist(image[:, :, 2].flatten(), bins=256, range=[0, 256], color='r', alpha=0.4, label="Red Channel")
axs[0].set_xlim([0, 256])
axs[0].legend()

# Gamma corrected image histograms (Red, Green, Blue in the same plot)
axs[1].set_title('Gamma Corrected Image - RGB Channels')
axs[1].hist(image_corrected[:, :, 0].flatten(), bins=256, range=[0, 256], color='b', alpha=0.4, label="Blue Channel")
axs[1].hist(image_corrected[:, :, 1].flatten(), bins=256, range=[0, 256], color='g', alpha=0.4, label="Green Channel")
axs[1].hist(image_corrected[:, :, 2].flatten(), bins=256, range=[0, 256], color='r', alpha=0.4, label="Red Channel")
axs[1].set_xlim([0, 256])
axs[1].set_ylim([0, 8000])  # Set y-axis limit to reduce dominance of zeros
axs[1].legend()

# Show the plots
plt.tight_layout()
plt.show()
