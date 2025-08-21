import cv2
import numpy as np

def analyze_image_colors(image_path):
    """
    分析图像中的颜色分布，特别是查找红色像素点
    """
    # 读取图像
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    print(f"图像尺寸: {image.shape}")
    print(f"图像数据类型: {image.dtype}")
    
    # 统计不同颜色的像素数量
    unique_colors, counts = np.unique(image.reshape(-1, image.shape[2]), axis=0, return_counts=True)
    
    print("\n图像中的所有颜色 (BGR格式):")
    sorted_colors = sorted(zip(unique_colors, counts), key=lambda x: x[1], reverse=True)
    
    for color, count in sorted_colors[:20]:  # 显示前20种最常见颜色
        print(f"颜色 {color} (BGR) 出现次数: {count}")
    
    # 查找接近红色的像素点
    # 红色在BGR格式中是 [0, 0, 255]
    print("\n查找红色像素点...")
    
    # 精确匹配红色
    red_pixels = np.sum(np.all(image == [0, 0, 255], axis=2))
    print(f"精确匹配红色 [0, 0, 255] 的像素数量: {red_pixels}")
    
    # 近似匹配红色 (考虑到可能有轻微的颜色变化)
    red_mask = (image[:,:,2] > 200) & (image[:,:,1] < 50) & (image[:,:,0] < 50)
    approx_red_pixels = np.sum(red_mask)
    print(f"近似红色像素数量: {approx_red_pixels}")
    
    # 如果有近似红色像素，显示一些样本
    if approx_red_pixels > 0:
        red_coords = np.where(red_mask)
        print(f"\n前10个近似红色像素点坐标 (y, x):")
        for i in range(min(10, len(red_coords[0]))):
            y, x = red_coords[0][i], red_coords[1][i]
            bgr_value = image[y, x]
            print(f"  坐标 ({x}, {y}): BGR {bgr_value}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = r"C:\wxh\mycode\gcellcode\result\N00007789\output_figure.jpg"
    
    print(f"分析图像: {image_path}")
    analyze_image_colors(image_path)