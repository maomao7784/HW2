import cv2
import numpy as np

def apply_histogram_equalization(input_image_path):
    loaded_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    histogram_counts = [0]*256
    for row_index in range(loaded_image.shape[0]):
        for column_index in range(loaded_image.shape[1]):
            histogram_counts[loaded_image[row_index, column_index]] += 1

    total_pixels = loaded_image.shape[0] * loaded_image.shape[1]
    cumulative_pixel_count, image_median = 0, 0
    for pixel_intensity in range(256):
        cumulative_pixel_count += histogram_counts[pixel_intensity]
        if cumulative_pixel_count >= total_pixels / 2:
            image_median = pixel_intensity
            break

    cumulative_density_function = [0]*256
    for pixel_intensity in range(256):
        if pixel_intensity <= image_median:
            cumulative_density_function[pixel_intensity] = sum(histogram_counts[:pixel_intensity+1])/sum(histogram_counts[:image_median+1])
        else:
            cumulative_density_function[pixel_intensity] = sum(histogram_counts[image_median+1:pixel_intensity+1])/sum(histogram_counts[image_median+1:])

    output_image = np.zeros_like(loaded_image)
    for row_index in range(loaded_image.shape[0]):
        for column_index in range(loaded_image.shape[1]):
            if loaded_image[row_index, column_index] <= image_median:
                output_image[row_index, column_index] = int(cumulative_density_function[loaded_image[row_index, column_index]] * image_median)
            else:
                output_image[row_index, column_index] = int(cumulative_density_function[loaded_image[row_index, column_index]] * (255 - image_median) + image_median)
    
    return output_image

output_directory = "/Users/linyinghsiao/Desktop/影像處理/HW2/Q4/"

if __name__ == "__main__":
    input_image_filename = 'aerialview-washedout.tif'
    processed_img = apply_histogram_equalization(input_image_filename)
    cv2.imwrite(output_directory + '44.tif', processed_img)
