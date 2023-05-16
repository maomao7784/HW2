import cv2
import numpy as np
output_folder = "/Users/linyinghsiao/Desktop/影像處理/HW2/Q5/"


def cvce_v1(image, window_size=7):
    height, width = image.shape
    half_window = window_size // 2
    padded_image = cv2.copyMakeBorder(image, half_window, half_window, half_window, half_window, cv2.BORDER_REFLECT)
    result_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            window = padded_image[y:y+window_size, x:x+window_size]
            hist, _ = np.histogram(window, bins=256, range=(0, 256))
            cdf = np.cumsum(hist)
            cdf_normalized = cdf / float(cdf[-1])
            result_image[y, x] = cdf_normalized[image[y, x]] * 255

    return result_image

image_path = "/Users/linyinghsiao/Desktop/影像處理/HW2/Q3/aerialview-washedout-fixed.tif"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# CVCE 1
enhanced_image = cvce_v1(image, window_size=7)

cv2.imwrite(output_folder + 'aerialview-washedout-fixed.tif', enhanced_image)
