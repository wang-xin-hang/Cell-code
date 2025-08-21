import numpy as np
import cv2
from skimage import measure
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
import os


@dataclass
class ProcessingConfig:
    """图像处理配置类，管理所有处理参数"""
    
    # 高斯模糊参数
    gaussian_kernel_size: Tuple[int, int] = (5, 5)
    gaussian_sigma: float = 5.0
    
    # 二值化参数
    binary_threshold: int = 40
    binary_max_value: int = 255
    threshold_type: int = cv2.THRESH_BINARY
    
    # 形态学操作参数
    dilate_kernel_size: Tuple[int, int] = (3, 3)
    dilate_iterations: int = 1
    erode_kernel_size: Tuple[int, int] = (3, 3)
    erode_iterations: int = 1
    
    # 边界清零参数
    boundary_width: int = 5
    
    # 连通域分析参数
    connectivity: int = 8
    skimage_connectivity: int = 2
    
    # 输出文件配置
    threshold_filename: str = "thres1.jpg"
    centers_filename: str = "centers.txt"
    output_dir: str = ""
    save_intermediate_results: bool = True


class ImagePreprocessor:
    """图像预处理器，负责灰度转换和高斯模糊"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """将彩色图像转换为灰度图像"""
        if len(image.shape) == 2:
            return image
        
        # 使用手写的加权平均方法（保持原有算法）
        return (0.2125 * image[:, :, 2] + 
                0.7154 * image[:, :, 0] + 
                0.0721 * image[:, :, 1]).astype(np.uint8)
    
    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """应用高斯模糊"""
        return cv2.GaussianBlur(
            image, 
            self.config.gaussian_kernel_size, 
            self.config.gaussian_sigma
        )
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """完整的预处理流程"""
        # 灰度转换
        gray_image = self.to_grayscale(image)
        # 高斯模糊
        blurred_image = self.apply_gaussian_blur(gray_image)
        return blurred_image


class ImageSegmentation:
    """图像分割器，负责二值化和形态学操作"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def binary_threshold(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """二值化处理"""
        ret, binary_image = cv2.threshold(
            image,
            self.config.binary_threshold,
            self.config.binary_max_value,
            self.config.threshold_type
        )
        return ret, binary_image
    
    def morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """形态学操作"""
        kernel_dilate = np.ones(self.config.dilate_kernel_size, np.uint8)
        dilated = cv2.dilate(image, kernel_dilate, iterations=self.config.dilate_iterations)
        return dilated
    
    def invert_and_boundary_clear(self, image: np.ndarray) -> np.ndarray:
        """图像取反和边界清零"""
        mask_inv = cv2.bitwise_not(image)
        
        # 边界清零
        width = self.config.boundary_width
        mask_inv[:, :width] = 0
        mask_inv[:, -width:] = 0
        mask_inv[:width, :] = 0
        mask_inv[-width:, :] = 0
        
        return mask_inv
    
    def segment(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """完整的分割流程"""
        ret, binary_image = self.binary_threshold(image)
        morphed_image = self.morphological_operations(binary_image)
        final_mask = self.invert_and_boundary_clear(morphed_image)
        return ret, final_mask


class ConnectedComponentAnalyzer:
    """连通域分析器，负责连通域分析和质心计算"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def analyze_with_skimage(self, mask: np.ndarray) -> Tuple[List[float], List[List[float]]]:
        """使用scikit-image进行连通域分析"""
        labels = measure.label(mask, connectivity=self.config.skimage_connectivity)
        properties = measure.regionprops(labels)
        
        areas = []
        centroids = []
        
        for i in range(np.max(labels)):
            if i < len(properties):
                areas.append(properties[i].area)
                # 坐标顺序 [y, x] -> [x, y]
                centroids.append([properties[i].centroid[1], properties[i].centroid[0]])
        
        return areas, centroids
    
    def analyze_with_opencv(self, mask: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """使用OpenCV进行连通域分析"""
        num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(
            mask, connectivity=self.config.connectivity
        )
        return num_labels, labels, stats, centers
    
    def analyze_components(self, mask: np.ndarray) -> Dict[str, Any]:
        """完整的连通域分析"""
        areas, centroids_skimage = self.analyze_with_skimage(mask)
        num_labels, labels, stats, centers_opencv = self.analyze_with_opencv(mask)
        
        results = {
            'skimage': {
                'areas': areas,
                'centroids': centroids_skimage,
                'count': len(centroids_skimage)
            },
            'opencv': {
                'num_labels': num_labels,
                'labels': labels,
                'stats': stats,
                'centers': centers_opencv,
                'count': num_labels - 1
            }
        }
        
        return results


def process_cell_image_simple(image_path: str, output_dir: str = "") -> Dict[str, Any]:
    """
    简化版细胞图像处理函数，只生成centers.txt和thres1.jpg文件
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录路径，默认为图像所在目录
        
    Returns:
        包含处理结果信息的字典
    """
    # 设置输出目录
    if not output_dir:
        output_dir = os.path.dirname(image_path) or "."
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置参数
    config = ProcessingConfig(
        output_dir=output_dir,
        save_intermediate_results=True
    )
    
    # 初始化处理组件
    preprocessor = ImagePreprocessor(config)
    segmentation = ImageSegmentation(config)
    analyzer = ConnectedComponentAnalyzer(config)
    
    # 读取图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"无法读取图像文件: {image_path}")
    
    # 预处理
    preprocessed = preprocessor.preprocess(original_image)
    
    # 分割
    threshold_value, mask = segmentation.segment(preprocessed)
    
    # 连通域分析
    analysis_results = analyzer.analyze_components(mask)
    
    # 保存centers.txt
    centers_path = os.path.join(output_dir, config.centers_filename)
    centroids = analysis_results['skimage']['centroids']
    if centroids:
        np.savetxt(centers_path, centroids)
    
    # 保存thres1.jpg (二值化结果)
    threshold_path = os.path.join(output_dir, config.threshold_filename)
    # 重新生成二值化图像用于保存
    _, binary_for_save = cv2.threshold(
        preprocessed, 
        config.binary_threshold, 
        config.binary_max_value, 
        config.threshold_type
    )
    cv2.imwrite(threshold_path, binary_for_save)
    
    # 返回处理结果
    results = {
        'cell_count': analysis_results['opencv']['count'],
        'centers_file': centers_path,
        'threshold_file': threshold_path,
        'analysis_results': analysis_results
    }
    
    return results


# 主程序入口
if __name__ == "__main__":
    try:
        # 使用简化函数处理图像
        results = process_cell_image_simple(
            r"C:\wxh\mycode\gcellcode\result\N00007789\N00007789_seg.jpg"
        )
        
        print(f"处理完成！发现 {results['cell_count']} 个细胞")
        print(f"质心坐标保存至: {results['centers_file']}")
        print(f"二值化图像保存至: {results['threshold_file']}")
        
    except FileNotFoundError as e:
        print(f"文件错误: {e}")
    except Exception as e:
        print(f"处理错误: {e}")