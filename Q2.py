import cv2
import numpy as np
output_folder = "/Users/linyinghsiao/Desktop/影像處理/HW2/Q2/"

image = cv2.imread('aerialview-washedout.tif', cv2.IMREAD_GRAYSCALE)

def linear_stretching(image, min_percentile=1, max_percentile=99):
    min_val = np.percentile(image, min_percentile)
    max_val = np.percentile(image, max_percentile)
    output_image = ((image - min_val) * (255 / (max_val - min_val))).astype('uint8')
    return output_image

enhanced_image = linear_stretching(image, min_percentile=5, max_percentile=99)

cv2.imwrite(output_folder +'aerialview-enhanced.tif', enhanced_image)
import numpy as np
import cv2

output_folder = "/Users/linyinghsiao/Desktop/影像處理/HW2/Q2/"

input_image = cv2.imread('einstein-low-contrast.tif', cv2.IMREAD_GRAYSCALE)

min_value = np.min(input_image)
max_value = np.max(input_image)

adjusted_image = (input_image - min_value) * (255.0 / (max_value - min_value))

adjusted_image = adjusted_image.astype(np.uint8)

cv2.imwrite(output_folder + 'Q2_image.tif', adjusted_image) 