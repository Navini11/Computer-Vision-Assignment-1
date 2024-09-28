import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the brain proton density image
image_path = 'a1images/a1images/brain_proton_density_slice.png'
brain_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

# Function to apply intensity transformation to accentuate white/gray matter
def accentuate_matter(image, matter='white'):
    if matter == 'white':
        # Apply a transformation to enhance white matter (e.g., intensify higher intensity values)
        transformed_image = np.interp(image, [0, 100, 150, 255], [0, 50, 200, 255])
    elif matter == 'gray':
        # Apply a transformation to enhance gray matter (e.g., intensify middle intensity values)
        transformed_image = np.interp(image, [0, 50, 120, 255], [0, 100, 180, 255])
    return transformed_image

# Apply the transformations for both white and gray matter
white_matter_image = accentuate_matter(brain_image, 'white')
gray_matter_image = accentuate_matter(brain_image, 'gray')

# Display original image
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(brain_image, cmap='gray')
plt.title('Original Brain Image')

# Display white matter accentuated image
plt.subplot(1, 3, 2)
plt.imshow(white_matter_image, cmap='gray')
plt.title('White Matter Accentuated')

# Display gray matter accentuated image
plt.subplot(1, 3, 3)
plt.imshow(gray_matter_image, cmap='gray')
plt.title('Gray Matter Accentuated')

plt.show()

# Plot the intensity transformations
plt.figure(figsize=(10, 5))
# Plot for white matter
plt.subplot(1, 2, 1)
plt.plot(np.linspace(0, 255, len(np.interp(np.arange(0, 256), [0, 100, 150, 255], [0, 50, 200, 255]))),
         np.interp(np.arange(0, 256), [0, 100, 150, 255], [0, 50, 200, 255]), color='blue')
plt.title('Intensity Transformation for White Matter')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')

# Plot for gray matter
plt.subplot(1, 2, 2)
plt.plot(np.linspace(0, 255, len(np.interp(np.arange(0, 256), [0, 50, 120, 255], [0, 100, 180, 255]))),
         np.interp(np.arange(0, 256), [0, 50, 120, 255], [0, 100, 180, 255]), color='green')
plt.title('Intensity Transformation for Gray Matter')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')

plt.show()

