#原始图片旋转、像素插值放大
import cv2
import numpy as np

def img_rotate(src, angel):
    h,w = src.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angel, 1.0)
    rotated_h = int((w * np.abs(M[0,1]) + (h * np.abs(M[0,0]))))
    rotated_w = int((h * np.abs(M[0,1]) + (w * np.abs(M[0,0]))))
    M[0,2] += (rotated_w - w) // 2
    M[1,2] += (rotated_h - h) // 2
    rotated_img = cv2.warpAffine(src, M, (rotated_w,rotated_h))

    return rotated_img

def rotate_and_resize_image(img_path,img_save_path ,angle=90, size=(448, 304)):
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return None
        
    img_rotated = img_rotate(img, angle)
    
    img_resized = cv2.resize(img_rotated, size, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(img_save_path, img_resized)
    return img_resized

if __name__ == "__main__":
    import sys
    import os
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "N00007789.jpg"
        if not os.path.exists(img_path):
            print("请提供图片路径作为参数，或确保N00007789.jpg在当前目录下")
            sys.exit(1)
            
    img_new = rotate_and_resize_image(img_path)
    if img_new is not None:
        cv2.imshow("img_new", img_new)
        cv2.waitKey(0)
        cv2.imwrite("img_new.jpg", img_new)