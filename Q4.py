import Q3
import cv2
import numpy as np
output_folder = "/Users/linyinghsiao/Desktop/影像處理/HW2/Q4/"

# 轉換為灰度圖像
image = cv2.imread('aerialview-washedout.tif', cv2.IMREAD_GRAYSCALE)

histogram, cdf, height, width = Q3.some_function()

median = np.median(image)

histogram1 = [0.0] * 256
for i in range(int(median)):
   histogram1[i] = histogram[i]

histogram2 = [0.0] * 256
for i in range(int(median), 256):
   histogram2[i] = histogram[i]

new_gray_values1 = [0.0] * 256
for i in range(int(median)):
   new_gray_values1[i] = round(cdf[i] * median)

new_gray_values2 = [0.0] * 256
for i in range(int(median), 256):
   new_gray_values2[i] = round(cdf[i] * (255 - median) + median)

for i in range(height):
   for j in range(width):
       if image[i][j] < median:
           image[i][j] = new_gray_values1[image[i][j]]
       else:
           image[i][j] = new_gray_values2[image[i][j]]

cv2.imwrite(output_folder + 'aerialview-washedout-fixed.tif', image)