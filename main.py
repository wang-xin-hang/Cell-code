import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Cell-Code'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Cell-Code', 'Seg_enhance'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Cell-Code-Deep-learning', 'models'))

from rotation_image import rotate_and_resize_image
import cv2
from Flipud_img import Flipud_img
from Seg_enhance import compare_and_save_images
from GetPoint import process_cell_image_simple
from Voronoi_Graph import generate_voronoi_diagram
from sobel import overlay_images
from Pixel_cropping import process_voronoi_image
from denseunet_test_main1 import process_images

def add_suffix(path, suffix):
    name, ext = os.path.splitext(path)
    new_path = name + suffix + ext
    return new_path

def main(img_path):
    if not os.path.exists(img_path):
        print(f"错误: 找不到输入图片 {img_path}")
        return False

    result_dir = "./result"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    img_name = os.path.basename(img_path)
    img_name = os.path.splitext(img_name)[0]
    result_subdir = os.path.join(result_dir, img_name)
    if not os.path.exists(result_subdir):
        os.mkdir(result_subdir)

    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return False
    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)
    rotated_filename = name + "_rotated" + ext
    rotated_path = os.path.join(result_subdir, rotated_filename)
    rotate_and_resize_image(img_path,rotated_path)
    print(f"图像处理完成，已保存为 {rotated_path}")


    fli_path = add_suffix(rotated_path, "_fli_1")
    try:
        Flipud_img(rotated_path, fli_path)
        print(f"图像处理完成，已保存为 {fli_path}")
    except ValueError as e:
        print(e)
        return False

    rotated_path_pre = add_suffix(rotated_path, "_pre")
    fli_path_pre = add_suffix(fli_path, "_pre")

    process_images(rotated_path, fli_path, rotated_path_pre, fli_path_pre)

    fli_path2 = add_suffix(fli_path_pre, "_fli_2")
    Flipud_img(fli_path_pre, fli_path2)
    seg_path = os.path.join(result_subdir, f"{img_name}_seg.jpg")
    compare_and_save_images(rotated_path_pre, fli_path2, seg_path)
    print(f"图像处理完成，已保存为 {seg_path}")

    try:
        results = process_cell_image_simple(seg_path)
        
        print(f"处理完成！发现 {results['cell_count']} 个细胞")
        print(f"质心坐标保存至: {results['centers_file']}")
        print(f"二值化图像保存至: {results['threshold_file']}")
        
    except FileNotFoundError as e:
        print(f"文件错误: {e}")
    except Exception as e:
        print(f"处理错误: {e}")

    
    thres1_path = os.path.join(result_subdir, "thres1.jpg")
    center_path = os.path.join(result_subdir, "centers.txt")
    json_path = os.path.join(result_subdir, f"{img_name}.json")
    output_path = os.path.join(result_subdir, "output_figure.png")
    generate_voronoi_diagram (thres1_path, center_path, json_path, output_path)

    vor_path = os.path.join(result_subdir, "Voronoi.jpg")
    process_voronoi_image(output_path, vor_path)

    sobel_path = os.path.join(result_subdir, "sobel.jpg")
    overlay_images(
        rotated_path,
        vor_path,
        sobel_path
    )

if __name__ == "__main__":
    img_path = r"N00007789.jpg"
    main(img_path)
