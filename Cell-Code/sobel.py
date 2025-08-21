import cv2
import numpy as np

def overlay_images(background_image_path, overlay_image_path, output_image_path):
    import cv2
    
    img1 = cv2.imread(background_image_path)
    img2 = cv2.imread(overlay_image_path)
    
    h, w = img1.shape[:2]
    img2 = cv2.resize(img2, (w, h))
    
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]
    
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
    
    mask_inv = cv2.bitwise_not(mask)
    
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)
    
    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst
    
    cv2.imwrite(output_image_path, img1)
    
    return img1

if __name__ == "__main__":
    result = overlay_images(
        r"G:\wxh\16ban\gcellcode\Cell-Code\img_fli.jpg",
        r"./Voronoi.jpg",
        "sobel.jpg"
    )