import cv2
import numpy as np
output_folder = "/Users/linyinghsiao/Desktop/影像處理/HW2/Q3/"

def process_image(image_name):
    # 轉換為灰度圖像
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

    # 獲取圖像的高度和寬度
    height, width = image.shape

    # 計算每個灰度級出現的次數
    histogram = [0.0] * 256
    for i in range(height):
        for j in range(width):
            histogram[image[i][j]] += 1

    # 計算每個灰度級的概率
    for i in range(len(histogram)):
        histogram[i] /= height * width

    # 計算每個灰度級的累積分佈概率
    cdf = [0.0] * 256
    cdf[0] = histogram[0]
    for i in range(1, len(histogram)):
        cdf[i] = cdf[i - 1] + histogram[i]

    # 計算每個灰度級的新灰度值
    new_gray_values = [0.0] * 256
    for i in range(len(cdf)):
        new_gray_values[i] = round(cdf[i] * 255)

    # 將原始圖像的灰度值替換為新的灰度值
    for i in range(height):
        for j in range(width):
            image[i][j] = new_gray_values[image[i][j]]

    return image

if __name__ == "__main__":
    image1_name = 'aerialview-washedout.tif'
    image2_name = 'einstein-low-contrast.tif'

    image1 = process_image(image1_name)
    image2 = process_image(image2_name)

    cv2.imwrite(output_folder + 'aerialview-washedout-fixed.tif', image1)
    cv2.imwrite(output_folder + 'einstein-low-contrast-fixed.tif', image2)
