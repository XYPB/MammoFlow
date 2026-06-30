import numpy as np
from matplotlib import pylab as pylab
from PIL import Image
import cv2
from skimage import io
from skimage import color


def enhance_contrast(image):
    # Convert the image to a NumPy array if it's a PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)  # Convert to 8-bit if not already

    # Apply histogram equalization
    enhanced_image = cv2.equalizeHist(image)

    return Image.fromarray(enhanced_image)


def right_orient_mammogram(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)  # Convert to 8-bit if not already
    left_nonzero = cv2.countNonZero(image[:, 0 : int(image.shape[1] / 2)])
    right_nonzero = cv2.countNonZero(image[:, int(image.shape[1] / 2) :])

    is_flipped = left_nonzero < right_nonzero
    if is_flipped:
        image = cv2.flip(image, 1)

    return image, is_flipped


def read_image(filename):
    image = io.imread(filename)
    image = color.rgb2gray(image)
    return image


def otsu_mask(img):
    median = np.median(img)
    _, thresh = cv2.threshold(img, median, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def otsu_cut(x):
    if isinstance(x, Image.Image):
        x = np.array(x)
    if x.max() <= 1:
        x = (x * 255).astype(np.uint8)  # Convert to 8-bit if not already
    mask = otsu_mask(x)
    # plt.imshow(mask, cmap='gray')
    # Convert to NumPy array if not already

    # Check if the matrix is empty or has no '1's
    if mask.size == 0 or not np.any(mask):
        return Image.fromarray(x)

    # Find the rows and columns where '1' appears
    rows = np.any(mask == 255, axis=1)
    cols = np.any(mask == 255, axis=0)

    # Find the indices of the rows and columns
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    # Crop and return the submatrix
    x = x[min_row : max_row + 1, min_col : max_col + 1]
    img = Image.fromarray(x)
    return img


def threshold_background(image, threshold=0):
    # Convert the image to a NumPy array if it's a PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)  # Convert to 8-bit if not already

    # Create a mask where pixel values are above the threshold
    mask = image > threshold

    # Create an output image initialized to black
    output_image = np.zeros_like(image)

    # Copy pixels from the original image where the mask is True
    output_image[mask] = image[mask]

    return Image.fromarray(output_image)


def remove_text_label(image, background_threshold=0):
    # Convert the image to a NumPy array if it's a PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)
    convert = False
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        convert = True
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)  # Convert to 8-bit if not already

    # Binarize the image using a naive non-zero thresholding
    binary_image = (image > background_threshold).astype(np.uint8) * 255

    # Apply Gaussian blur to the binarized image
    blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 2.0)
    # Binarize the blurred image again
    binary_image = (blurred_image > background_threshold).astype(np.uint8) * 255
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )

    # Create an output image to store the result
    output_image = image.copy()

    # Remove small connected components
    for i in range(1, num_labels):  # Start from 1 to skip the background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 1e4 and area < np.max(
            stats[:, cv2.CC_STAT_AREA]
        ):  # Threshold for small areas, adjust as needed
            x, y, w, h = (
                stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_HEIGHT],
            )
            output_image[y : y + h, x : x + w] = 0  # Set the region to black
    # if image is set to pure black, return the original image
    if np.all(output_image == 0):
        output_image = image
    if convert:
        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2RGB)
    return output_image


def gaussian_blur(image):
    # Convert the image to a NumPy array if it's a PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)  # Convert to 8-bit if not already

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(
        image,
        (
            5,
            5,
        ),
        10,
    )

    return Image.fromarray(blurred_image)
