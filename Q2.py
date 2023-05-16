import cv2
import numpy as np
output_folder = "/Users/linyinghsiao/Desktop/影像處理/HW2/Q2/"

image = cv2.imread('aerialview-washedout.tif', cv2.IMREAD_GRAYSCALE)

def linear_stretching(image, min_percentile=1, max_percentile=99):
    min_val = np.percentile(image, min_percentile)
    max_val = np.percentile(image, max_percentile)
    output_image = ((image - min_val) * (255 / (max_val - min_val))).astype('uint8')
    return output_image

# 應用線性拉伸以增強對比度
enhanced_image = linear_stretching(image, min_percentile=5, max_percentile=99)

cv2.imwrite(output_folder +'aerialview-enhanced.tif', enhanced_image)
