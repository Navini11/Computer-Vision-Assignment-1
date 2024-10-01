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

