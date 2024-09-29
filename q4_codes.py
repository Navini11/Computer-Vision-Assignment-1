import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = r'a1images\a1images\spider.png'
image = cv2.imread(image_path)

# Convert image from BGR to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Split the HSV channels
hue, saturation, value = cv2.split(hsv_image)

# Plot the Hue, Saturation, and Value channels
plt.figure(figsize=(12, 4))

# Plot Hue channel
plt.subplot(1, 3, 1)
plt.imshow(hue, cmap='hsv')
plt.title("Hue Channel")
plt.axis('off')

# Plot Saturation channel
plt.subplot(1, 3, 2)
plt.imshow(saturation, cmap='gray')
plt.title("Saturation Channel")
plt.axis('off')

# Plot Value channel
plt.subplot(1, 3, 3)
plt.imshow(value, cmap='gray')
plt.title("Value Channel")
plt.axis('off')

plt.tight_layout()
plt.show()

# Intensity transformation function
def intensity_transform(x, a, sigma=70):
    x = np.clip(x, 0, 255)  # Ensure x stays within valid range
    transformed = x + a * 128 * np.exp(-(x - 128) ** 2 / (2 * sigma ** 2))
    return np.clip(transformed, 0, 255).astype(np.uint8)

# Apply intensity transformation to the saturation plane
a = 0.4  # Selected a value
transformed_saturation = intensity_transform(saturation, a)

# Recombine the modified saturation with the hue and value planes
transformed_hsv = cv2.merge([hue, transformed_saturation, value])

# Convert back to BGR color space
vibrance_enhanced_image = cv2.cvtColor(transformed_hsv, cv2.COLOR_HSV2BGR)

# Display the original image, vibrance-enhanced image, and saturation transformation
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(vibrance_enhanced_image, cv2.COLOR_BGR2RGB))
plt.title(f"Vibrance Enhanced Image (a={a})")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(transformed_saturation, cmap='gray')
plt.title("Transformed Saturation")
plt.axis('off')

plt.tight_layout()
plt.show()



## code for finding optimal a value by plotting for a=0 to a=1 by step of 0.1
for i in range (0,11,1):
    a=i/10

    transformed_saturation = intensity_transform(saturation, a)

    # Recombine the modified saturation with the hue and value planes
    transformed_hsv = cv2.merge([hue, transformed_saturation, value])

    # Convert back to BGR color space
    vibrance_enhanced_image = cv2.cvtColor(transformed_hsv, cv2.COLOR_HSV2BGR)

    # Display the original image, vibrance-enhanced image, and saturation transformation
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(vibrance_enhanced_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Vibrance Enhanced Image (a={a})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(transformed_saturation, cmap='gray')
    plt.title("Transformed Saturation")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


