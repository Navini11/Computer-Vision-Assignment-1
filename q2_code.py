import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the brain proton density image
image_path = 'a1images/a1images/brain_proton_density_slice.png'
brain_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
## selected pixel ranges
white_th = np.array([120,175], dtype=np.uint8)
grey_th = np.array([180,240], dtype=np.uint8)

white_matter_transform = np.zeros(256, dtype=np.uint8)
white_matter_transform[white_th[0]:white_th[1]+1] = np.linspace(white_th[0], white_th[1],white_th[1]-white_th[0]+1,dtype=np.uint8)

grey_matter_transform = np.zeros(256, dtype=np.uint8)
grey_matter_transform[grey_th[0]:grey_th[1]+1] = np.linspace(grey_th[0], grey_th[1],grey_th[1]-grey_th[0]+1,dtype=np.uint8)

white_matter = cv2.LUT(brain_image, white_matter_transform)
grey_matter = cv2.LUT(brain_image, grey_matter_transform)

#plot intensity transform functions
fig,ax = plt.subplots(1,2,figsize=(12,4))

ax[0].plot(white_matter_transform, 'r')
ax[0].set_title('White Matter Intensity Transform')
ax[0].set_xlabel('Intensity Level')
ax[0].set_ylabel('Pixel Value')
ax[0].set_ylim([0,255])

ax[1].plot(grey_matter_transform, 'r')
ax[1].set_title('Grey Matter Intensity Transform')
ax[1].set_xlabel('Intensity Level')
ax[1].set_ylabel('Pixel Value')
ax[1].set_ylim([0,255])

#plot images

fig,ax = plt.subplots(1,3,figsize=(12,4))

ax[0].imshow(brain_image, cmap='gray')
ax[0].set_title('Original Image')

ax[1].imshow(white_matter, cmap='gray')
ax[1].set_title('White Matter')

ax[2].imshow(grey_matter, cmap='gray')
ax[2].set_title('Grey Matter')

plt.show()
