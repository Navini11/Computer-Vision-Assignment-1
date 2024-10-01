def zoom_image(image, factor, method='nearest'):
    """Zoom the image by a given factor using the specified interpolation method."""
    newsize = (int(image.shape[1] * factor), int(image.shape[0] * factor))
    if method == 'nearest':
        zoomed_image = cv2.resize(image, newsize, interpolation=cv2.INTER_NEAREST)
    elif method == 'bilinear':
        zoomed_image = cv2.resize(image, newsize, interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError("Method must be 'nearest' or 'bilinear'.")
    return zoomed_image

def compute_ssd(original, zoomed):
    ssd = np.sum((original - zoomed) ** 2)
    normalized_ssd = ssd / original.size
    return normalized_ssd

# Display images
def display_images(original, zoomed_nearest, zoomed_bilinear):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(zoomed_nearest, cv2.COLOR_BGR2RGB))
    plt.title(f'Nearest Neighbour Method')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(zoomed_bilinear, cv2.COLOR_BGR2RGB))
    plt.title(f'Bilinear Interpolation Method')
    plt.show()

#################################################image 1##################################

# Load images
large_image_path1 = r'a1images\a1images\a1q5images\im01.png'
small_image_path1= r'a1images\a1images\a1q5images\im01small.png'

# Load original and zoomed-out images for testing
large_image1 = cv2.imread(large_image_path1)
small_image1 = cv2.imread(small_image_path1)
# Zoom the small image by a factor of 4
factor = 4
zoomed_image_nearest1 = zoom_image(small_image1, factor, method='nearest')
zoomed_image_bilinear1 = zoom_image(small_image1, factor, method='bilinear')

# Compute SSD with the original large image
ssd_nearest1 = compute_ssd(large_image1, zoomed_image_nearest1)
ssd_bilinear1 = compute_ssd(large_image1, zoomed_image_bilinear1)

display_images(large_image1, zoomed_image_nearest1,zoomed_image_bilinear1 )

print('Nearest neighbour ssd = ', ssd_nearest1)
print('Bilinear interpolation ssd = ', ssd_bilinear1)

#####################################################image 2############################

large_image_path2 = r'a1images\a1images\a1q5images\im02.png'
small_image_path2 = r'a1images\a1images\a1q5images\im02small.png'

# Load original and zoomed-out images for testing
large_image2 = cv2.imread(large_image_path2)
small_image2 = cv2.imread(small_image_path2)
# Zoom the small image by a factor of 4
factor = 4
zoomed_image_nearest2 = zoom_image(small_image2, factor, method='nearest')
zoomed_image_bilinear2 = zoom_image(small_image2, factor, method='bilinear')

# Compute SSD with the original large image
ssd_nearest2 = compute_ssd(large_image2, zoomed_image_nearest2)
ssd_bilinear2 = compute_ssd(large_image2, zoomed_image_bilinear2)

display_images(large_image2, zoomed_image_nearest2,zoomed_image_bilinear2 )

print('Nearest neighbour ssd = ', ssd_nearest2)
print('Bilinear interpolation ssd = ', ssd_bilinear2)


##################################################image 3#################################

# Load images
large_image_path3 = r'a1images\a1images\a1q5images\im03.png'
small_image_path3 = r'a1images\a1images\a1q5images\im03small.png'

# Load original and zoomed-out images for testing
large_image3 = cv2.imread(large_image_path3)
small_image3 = cv2.imread(small_image_path3)

def resize_to_match(image, target_shape):
    return cv2.resize(image, (target_shape[1], target_shape[0]))

# Zoom the small image by a factor of 4
factor = 4
zoomed_image_nearest3 = zoom_image(small_image3, factor, method='nearest')
zoomed_image_bilinear3 = zoom_image(small_image3, factor, method='bilinear')

# Resize the zoomed images to match the original large image dimensions
zoomed_image_nearest3_resized = resize_to_match(zoomed_image_nearest3, large_image3.shape)
zoomed_image_bilinear3_resized = resize_to_match(zoomed_image_bilinear3, large_image3.shape)

# Now compute SSD with the resized images
ssd_nearest3 = compute_ssd(large_image3, zoomed_image_nearest3_resized)
ssd_bilinear3 = compute_ssd(large_image3, zoomed_image_bilinear3_resized)

# Display images
display_images(large_image3, zoomed_image_nearest3_resized, zoomed_image_bilinear3_resized)

print('Nearest neighbour ssd = ', ssd_nearest3)
print('Bilinear interpolation ssd = ', ssd_bilinear3)

###################################################image 4#################################

# Load images
large_image_path4 = r'a1images\a1images\a1q5images\taylor.jpg'
small_image_path4 = r'a1images\a1images\a1q5images\taylor_small.jpg'

# Load original and zoomed-out images for testing
large_image4 = cv2.imread(large_image_path4)
small_image4 = cv2.imread(small_image_path4)

def resize_to_match(image, target_shape):
    return cv2.resize(image, (target_shape[1], target_shape[0]))

# Zoom the small image by a factor of 4
factor = 4
zoomed_image_nearest4 = zoom_image(small_image4, factor, method='nearest')
zoomed_image_bilinear4 = zoom_image(small_image4, factor, method='bilinear')

# Resize the zoomed images to match the original large image dimensions
zoomed_image_nearest4_resized = resize_to_match(zoomed_image_nearest4, large_image4.shape)
zoomed_image_bilinear4_resized = resize_to_match(zoomed_image_bilinear4, large_image4.shape)

# Now compute SSD with the resized images
ssd_nearest4 = compute_ssd(large_image4, zoomed_image_nearest4_resized)
ssd_bilinear4 = compute_ssd(large_image4, zoomed_image_bilinear4_resized)

# Display images
display_images(large_image4, zoomed_image_nearest4_resized, zoomed_image_bilinear4_resized)

print('Nearest neighbour ssd = ', ssd_nearest4)
print('Bilinear interpolation ssd = ', ssd_bilinear4)








