import cv2
import numpy as np

def local_histogram(input_img, window):
    val_min = input_img.min()
    val_max = input_img.max()
    h, w = input_img.shape
    histogram = []
    window_half = int((window-1) / 2)
    for gray_scale in range(256):
        emp = []
        mask = np.where(input_img == gray_scale)
        for intensity in range(256):
            hist = 0
            wght = np.abs(gray_scale - intensity + 1) / (val_max - val_min + 1) 
            for row, col in zip(mask[0], mask[1]):
                if row >= window_half and row <= h - (window_half+1) and col >= window_half and col <= w - (window_half+1):
                    hist += (input_img[(row- window_half):(row + window_half+1), (col - window_half):(col + window_half+1)] == intensity).sum()
            emp.append(hist*wght )
        histogram.append(emp)
    return histogram

def cvce(input_img, window):
    h, w = input_img.shape
    local_hist = local_histogram(input_img, window)

    total_val = 0
    for gray_scale in range(256):
        for intensity in range(256):
            total_val += local_hist[gray_scale][intensity]

    pdf_arr = []
    for gray_scale in range(256):
        pdf_val = 0
        for g in range(gray_scale):
            for i in range(gray_scale):
                pdf_val += local_hist[g][i]
        pdf_arr.append(pdf_val / total_val)

    cdf_uniform = []
    for gray_scale in range(256):
        cdf_uniform.append((gray_scale + 1)**2 / (256**2))

    cdf_index = []
    for gray_scale in range(256):
        cdf_index.append((np.abs(np.array(pdf_arr[gray_scale]) - np.array(cdf_uniform))).argmin())
        
    output_img = np.empty((h,w))
    for gray_scale in range(256):
        mask = np.where(input_img == gray_scale)
        output_img[mask[0], mask[1]] = cdf_index[gray_scale]
    output_img = output_img.astype(np.uint8)

    return output_img

img_path = "/Users/linyinghsiao/Desktop/影像處理/HW2/einstein-low-contrast.tif"
output_folder = "/Users/linyinghsiao/Desktop/影像處理/HW2/Q5/"

if __name__ == "__main__":
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    ce_img = cvce(img,7)
    cv2.imwrite(output_folder + '6.tif', ce_img)
