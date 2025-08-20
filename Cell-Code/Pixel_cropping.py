#裁剪维诺图
import cv2
import numpy as np

def find_first_red_pixel(image):
    """
    找到图像中第一个红色像素点的坐标
    """
    height, width, _ = image.shape
    for y in range(height):
        for x in range(width):
            pixel = image[y, x]
            if np.array_equal(pixel, [0, 0, 255]):  # 红色像素点的BGR值为 [0, 0, 255]
                return x, y

def find_last_red_pixel(image):
    """
    找到图像中最后一个红色像素点的坐标
    """
    height, width, _ = image.shape
    for y in range(height-1, -1, -1):
        for x in range(width-1, -1, -1):
            pixel = image[y, x]
            if np.array_equal(pixel, [0, 0, 255]):  # 红色像素点的BGR值为 [0, 0, 255]
                return x, y

def crop_image(image, start_coord, end_coord):
    """
    裁剪图像，从起始坐标到结束坐标
    """
    return image[start_coord[1]:end_coord[1]+1, start_coord[0]:end_coord[0]+1]

# 读取图片
image = cv2.imread('./output_figure.png')

# 找到第一个红色像素点和最后一个红色像素点
start_coord = find_first_red_pixel(image)
end_coord = find_last_red_pixel(image)

# 裁剪图像
cropped_image = crop_image(image, start_coord, end_coord)
cv2.imwrite("Voronoi.jpg", cropped_image)
# 显示原始图像和裁剪后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
