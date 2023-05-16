import cv2
import numpy as np
output_folder = "/Users/linyinghsiao/Desktop/影像處理/HW2/Q1/"

# 轉換為灰度圖像
image = cv2.imread('text-broken.tif', cv2.IMREAD_GRAYSCALE)

# 二值化圖像
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# 創建一個結構元素
kernel = np.ones((3, 3), np.uint8)

# 對二值圖像進行膨脹操作
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

edges = cv2.Canny(dilated_image, 100, 200)

cv2.imwrite(output_folder + 'text-fixed.tif', dilated_image)
cv2.imwrite(output_folder + 'character_boundaries.tif', edges)
