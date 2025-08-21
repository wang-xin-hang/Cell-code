#裁剪维诺图
import cv2
import numpy as np

def find_first_red_pixel(image):
    height, width, _ = image.shape
    for y in range(height):
        for x in range(width):
            pixel = image[y, x]
            if np.array_equal(pixel, [0, 0, 255]):
                return x, y

def find_last_red_pixel(image):
    height, width, _ = image.shape
    for y in range(height-1, -1, -1):
        for x in range(width-1, -1, -1):
            pixel = image[y, x]
            if np.array_equal(pixel, [0, 0, 255]):
                return x, y

def crop_image(image, start_coord, end_coord):
    return image[start_coord[1]:end_coord[1]+1, start_coord[0]:end_coord[0]+1]

def process_voronoi_image(imagepath,savepath):
    image = cv2.imread(imagepath)
    start_coord = find_first_red_pixel(image)
    end_coord = find_last_red_pixel(image)
    cropped_image = crop_image(image, start_coord, end_coord)
    cv2.imwrite(savepath, cropped_image)

if __name__ == '__main__':
    imagepath = r'C:\wxh\mycode\gcellcode\result\N00007789\output_figure.png'
    savepath = r'C:\wxh\mycode\gcellcode\result\N00007789\output_figure1.png'
    process_voronoi_image(imagepath,savepath)