# 图像增强算法，图像锐化算法
# 1）基于直方图均衡化 2）基于拉普拉斯算子 3）基于对数变换 4）基于伽马变换 5)CLAHE 6)retinex-SSR 7)retinex-MSR
# 其中，基于拉普拉斯算子的图像增强为利用空域卷积运算实现滤波
# 基于同一图像对比增强效果
# 直方图均衡化:对比度较低的图像适合使用直方图均衡化方法来增强图像细节
# 拉普拉斯算子可以增强局部的图像对比度
# log对数变换对于整体对比度偏低并且灰度值偏低的图像增强效果较好
# 伽马变换对于图像对比度偏低，并且整体亮度值偏高（对于相机过曝）情况下的图像增强效果明显

import cv2
import numpy as np
import matplotlib.pyplot as plt





# 直方图均衡增强
def hist(image):
    r, g, b = cv2.split(image)
    r1 = cv2.equalizeHist(r)
    g1 = cv2.equalizeHist(g)
    b1 = cv2.equalizeHist(b)
    image_equal_clo = cv2.merge([r1, g1, b1])
    return image_equal_clo


# 拉普拉斯算子
def laplacian(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image_lap = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    return image_lap


# 对数变换
def log(image):
    image_log = np.uint8(np.log(np.array(image) + 1))
    cv2.normalize(image_log, image_log, 0, 255, cv2.NORM_MINMAX)
    # 转换成8bit图像显示
    cv2.convertScaleAbs(image_log, image_log)
    return image_log


# 伽马变换
def gamma(image):
    fgamma = 2
    image_gamma = np.uint8(np.power((np.array(image) / 255.0), fgamma) * 255.0)
    cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
    cv2.convertScaleAbs(image_gamma, image_gamma)
    return image_gamma


# 限制对比度自适应直方图均衡化CLAHE
def clahe(image):
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(9, 9))
    # ret = cv2.equalizeHist(image)
    # cv2.imshow("ret", ret)
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image_clahe = cv2.merge([b, g, r])
    # cv2.imshow("image_clahe", Gray)
    return image_clahe


def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


# retinex SSR
def SSR(src_img, size):
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    dst_Img = cv2.log(img/255.0)
    dst_Lblur = cv2.log(L_blur/255.0)
    dst_IxL = cv2.multiply(dst_Img, dst_Lblur)
    log_R = cv2.subtract(dst_Img, dst_IxL)

    dst_R = cv2.normalize(log_R,None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8


def SSR_image(image):
    size = 3
    b_gray, g_gray, r_gray = cv2.split(image)
    b_gray = SSR(b_gray, size)
    g_gray = SSR(g_gray, size)
    r_gray = SSR(r_gray, size)
    result = cv2.merge([b_gray, g_gray, r_gray])
    return result


# retinex MMR
def MSR(img, scales):
    weight = 1 / 3.0
    scales_size = len(scales)
    h, w = img.shape[:2]
    log_R = np.zeros((h, w), dtype=np.float32)

    for i in range(scales_size):
        img = replaceZeroes(img)
        L_blur = cv2.GaussianBlur(img, (scales[i], scales[i]), 0)
        L_blur = replaceZeroes(L_blur)
        dst_Img = cv2.log(img/255.0)
        dst_Lblur = cv2.log(L_blur/255.0)
        dst_Ixl = cv2.multiply(dst_Img, dst_Lblur)
        log_R += weight * cv2.subtract(dst_Img, dst_Ixl)

    dst_R = cv2.normalize(log_R,None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8


def MSR_image(image):
    scales = [15, 101, 301]  # [3,5,9]
    b_gray, g_gray, r_gray = cv2.split(image)
    b_gray = MSR(b_gray, scales)
    g_gray = MSR(g_gray, scales)
    r_gray = MSR(r_gray, scales)
    result = cv2.merge([b_gray, g_gray, r_gray])
    return result


if __name__ == "__main__":
    # image = cv2.imread(r"E:\111\datas\jpg\7.jpg")
    # 原图
    img1 = cv2.imread(r"Seg_enhance/Test_other_images/N00007788/N00007788_448x304.jpg")
    # 掩膜
    img2 = cv2.imread(r"./Voronoi.jpg")

    h, w = img1.shape[:2]
    img2 = cv2.resize(img2, (w, h))
    # 220, 280

    # I want to put logo on top-left corner, So I create a ROI
    # 首先获取原始图像roi
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]
    print(img1.shape)

    # 原始图像转化为灰度值
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('img2gray', img2gray)
    cv2.waitKey(0)
    '''
    将一个灰色的图片，变成要么是白色要么就是黑色。（大于规定thresh值就是设置的最大值（常为255，也就是白色））
    '''
    # 将灰度值二值化，得到ROI区域掩模
    ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)

    # cv2.imshow('mask', mask)
    cv2.waitKey(0)

    # ROI掩模区域反向掩模
    mask_inv = cv2.bitwise_not(mask)

    # cv2.imshow('mask_inv', mask_inv)
    cv2.waitKey(0)

    # 掩模显示背景
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    # img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # cv2.imshow('img1_bg', img1_bg)
    cv2.waitKey(0)

    # 掩模显示前景
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)
    # img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    # cv2.imshow('img2_fg', img2_fg)
    cv2.waitKey(0)

    # 前背景图像叠加
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst

    # cv2.imshow('res1', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img1[0:rows, 0:cols] = dst

    cv2.imshow('res2', img1)
    cv2.imwrite("sobel.jpg", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


