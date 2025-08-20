#原始图片旋转、像素插值放大
import cv2
import numpy as np


if __name__ == "__main__":
    # 读取图像
    img_path = "./Seg_enhance/Test_other_images/N00007788/N00007788_X_fli_pre.jpg"
    img = cv2.imread(img_path)
    img_fli = np.flipud(img)
    cv2.imshow("img_new", img_fli)
    cv2.waitKey(0)
    cv2.imwrite("img_fli.jpg", img_fli)