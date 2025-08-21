import cv2
import numpy as np

def Flipud_img(img_path, fli_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    img_fli = np.flipud(img)

    cv2.imwrite(fli_path, img_fli)
    return True

if __name__ == "__main__":
    import sys
    import os
    if len(sys.argv) > 2:
        img_path = sys.argv[1]
        fli_path = sys.argv[2]
    else:
        img_path = "img_new.jpg"
        fli_path = "img_new_fli.jpg"
        if not os.path.exists(img_path):
            print("请提供输入和输出图片路径作为参数，或确保img_new.jpg在当前目录下")
            sys.exit(1)
    
    try:
        Flipud_img(img_path, fli_path)
        print(f"图像已保存到: {fli_path}")
    except ValueError as e:
        print(e)
        sys.exit(1)