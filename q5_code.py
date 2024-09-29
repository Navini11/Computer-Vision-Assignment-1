import cv2
import numpy as np
import matplotlib.pyplot as plt

# Custom function for histogram equalization
def custom_histogram_equalization(img):
    # Get the image histogram
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    
    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    
    # Normalize the CDF
    cdf_normalized = cdf * hist.max() / cdf.max()
    
    # Apply histogram equalization formula
    cdf_m = np.ma.masked_equal(cdf, 0)  # Mask zeros
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())  # Scale
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')  # Fill masked values with 0
    
    # Map the original image pixels using the equalized CDF
    equalized_img = cdf_final[img]
    
    return equalized_img

# Load the image and convert to grayscale
image_path = r'a1images\a1images\spider.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply custom histogram equalization
equalized_image = custom_histogram_equalization(image)

# Function to plot histograms
def plot_histograms(original_img, equalized_img):
    # Plot histograms
    plt.figure(figsize=(12, 6))
    
    # Original image histogram
    plt.subplot(2, 2, 1)
    plt.hist(original_img.flatten(), 256, [0, 256], color='gray')
    plt.title("Original Histogram")
    
    # Equalized image histogram
    plt.subplot(2, 2, 2)
    plt.hist(equalized_img.flatten(), 256, [0, 256], color='gray')
    plt.title("Equalized Histogram")
    
    # Original image
    plt.subplot(2, 2, 3)
    plt.imshow(original_img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    # Equalized image
    plt.subplot(2, 2, 4)
    plt.imshow(equalized_img, cmap='gray')
    plt.title("Equalized Image")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Plot histograms before and after equalization
plot_histograms(image, equalized_image)
