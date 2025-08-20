#原始图片旋转、像素插值放大
import cv2
import numpy as np

def img_rotate(src, angel):
    """逆时针旋转图像任意角度

    Args:
        src (np.array): [原始图像]
        angel (int): [逆时针旋转的角度]

    Returns:
        [array]: [旋转后的图像]
    """
    h,w = src.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angel, 1.0)
    # 调整旋转后的图像长宽
    rotated_h = int((w * np.abs(M[0,1]) + (h * np.abs(M[0,0]))))
    rotated_w = int((h * np.abs(M[0,1]) + (w * np.abs(M[0,0]))))
    M[0,2] += (rotated_w - w) // 2
    M[1,2] += (rotated_h - h) // 2
    # 旋转图像
    rotated_img = cv2.warpAffine(src, M, (rotated_w,rotated_h))

    return rotated_img

if __name__ == "__main__":
    # 读取图像
    img_path = "Seg_enhance/Test_other_images/N00007788/N00007788.jpg"
    img = cv2.imread(img_path)
    img_rotated = img_rotate(img, 90)
    crop_size = (448, 304)
    img_new = cv2.resize(img_rotated, crop_size, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("img_new", img_new)
    cv2.waitKey(0)
    cv2.imwrite("img_new.jpg", img_new)