
import numpy as np
import cv2
from skimage import measure
import logging
from typing import Tuple, List, Optional, Dict, Any
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
    
    # 可视化参数
    result_alpha: float = 0.8
    overlay_alpha: float = 0.5
    marker_color: Tuple[int, int, int] = (0, 0, 255)
    marker_type: int = 1
    marker_size: int = 2
    marker_thickness: int = 2
    text_color: Tuple[int, int, int] = (0, 255, 0)
    text_thickness: int = 2
    text_font_scale: float = 0.75
    
    # 输出文件配置
    save_intermediate_results: bool = True
    output_dir: str = ""
    threshold_filename: str = "thres1.jpg"
    mask_filename: str = "mask_inv.txt"
    centers_filename: str = "centers.txt"


class ImagePreprocessor:
    """图像预处理器，负责灰度转换和高斯模糊"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def to_grayscale(self, image: np.ndarray, method: str = "opencv") -> np.ndarray:
        """
        将彩色图像转换为灰度图像
        
        Args:
            image: 输入的彩色图像
            method: 转换方法，"opencv"使用cv2.cvtColor，"manual"使用手写加权平均
            
        Returns:
            灰度图像
        """
        if len(image.shape) == 2:
            return image
            
        if method == "opencv":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif method == "manual":
            # 保持原有的手写加权平均方法以确保兼容性
            return self._manual_grayscale(image)
        else:
            raise ValueError(f"不支持的转换方法: {method}")
    
    def _manual_grayscale(self, image: np.ndarray) -> np.ndarray:
        """手写的加权平均灰度转换（保持原有算法）"""
        h, w = image.shape[:2]
        gray = np.zeros((h, w), dtype=np.uint8)
        
        # 使用向量化操作替代双重循环，提高性能
        gray = (0.2125 * image[:, :, 2] + 
                0.7154 * image[:, :, 0] + 
                0.0721 * image[:, :, 1]).astype(np.uint8)
        
        return gray
    
    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """应用高斯模糊"""
        return cv2.GaussianBlur(
            image, 
            self.config.gaussian_kernel_size, 
            self.config.gaussian_sigma
        )
    
    def preprocess(self, image: np.ndarray, grayscale_method: str = "manual") -> np.ndarray:
        """完整的预处理流程"""
        self.logger.info("开始图像预处理")
        
        # 灰度转换
        gray_image = self.to_grayscale(image, grayscale_method)
        self.logger.info(f"灰度转换完成，使用方法: {grayscale_method}")
        
        # 高斯模糊
        blurred_image = self.apply_gaussian_blur(gray_image)
        self.logger.info("高斯模糊处理完成")
        
        return blurred_image


class ImageSegmentation:
    """图像分割器，负责二值化和形态学操作"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def binary_threshold(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """二值化处理"""
        ret, binary_image = cv2.threshold(
            image,
            self.config.binary_threshold,
            self.config.binary_max_value,
            self.config.threshold_type
        )
        self.logger.info(f"二值化完成，阈值: {ret}")
        return ret, binary_image
    
    def morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """形态学操作"""
        # 膨胀操作
        kernel_dilate = np.ones(self.config.dilate_kernel_size, np.uint8)
        dilated = cv2.dilate(image, kernel_dilate, iterations=self.config.dilate_iterations)
        self.logger.info("膨胀操作完成")
        
        return dilated
    
    def invert_and_boundary_clear(self, image: np.ndarray) -> np.ndarray:
        """图像取反和边界清零"""
        # 取反
        mask_inv = cv2.bitwise_not(image)
        
        # 边界清零
        width = self.config.boundary_width
        mask_inv[:, :width] = 0  # 左边界
        mask_inv[:, -width:] = 0  # 右边界
        mask_inv[:width, :] = 0  # 上边界
        mask_inv[-width:, :] = 0  # 下边界
        
        self.logger.info("图像取反和边界清零完成")
        return mask_inv
    
    def segment(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """完整的分割流程"""
        self.logger.info("开始图像分割")
        
        # 二值化
        ret, binary_image = self.binary_threshold(image)
        
        # 形态学操作
        morphed_image = self.morphological_operations(binary_image)
        
        # 取反和边界处理
        final_mask = self.invert_and_boundary_clear(morphed_image)
        
        self.logger.info("图像分割完成")
        return ret, final_mask


class ConnectedComponentAnalyzer:
    """连通域分析器，负责连通域分析和质心计算"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_with_skimage(self, mask: np.ndarray) -> Tuple[List[float], List[List[float]]]:
        """使用scikit-image进行连通域分析"""
        labels = measure.label(mask, connectivity=self.config.skimage_connectivity)
        properties = measure.regionprops(labels)
        
        areas = []
        centroids = []
        
        for i in range(np.max(labels)):
            if i < len(properties):
                areas.append(properties[i].area)
                # 注意：这里保持原有的坐标顺序 [y, x] -> [x, y]
                centroids.append([properties[i].centroid[1], properties[i].centroid[0]])
        
        self.logger.info(f"scikit-image分析完成，发现 {len(centroids)} 个连通域")
        return areas, centroids
    
    def analyze_with_opencv(self, mask: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """使用OpenCV进行连通域分析"""
        num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(
            mask, connectivity=self.config.connectivity
        )
        
        self.logger.info(f"OpenCV分析完成，发现 {num_labels-1} 个连通域")
        return num_labels, labels, stats, centers
    
    def analyze_components(self, mask: np.ndarray) -> Dict[str, Any]:
        """完整的连通域分析"""
        self.logger.info("开始连通域分析")
        
        # 使用两种方法进行分析（保持原有逻辑）
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
                'count': num_labels - 1  # 减去背景
            }
        }
        
        self.logger.info("连通域分析完成")
        return results


class ResultVisualizer:
    """结果可视化器，负责结果展示和保存"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_colored_components(self, original_image: np.ndarray, labels: np.ndarray, 
                                num_labels: int) -> np.ndarray:
        """为不同连通域分配随机颜色"""
        output = np.zeros((original_image.shape[0], original_image.shape[1], 3), np.uint8)
        
        for i in range(1, num_labels):
            mask = labels == i
            output[:, :, 0][mask] = np.random.randint(0, 255)
            output[:, :, 1][mask] = np.random.randint(0, 255)
            output[:, :, 2][mask] = np.random.randint(0, 255)
        
        return output
    
    def draw_centroids(self, image: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """在图像上绘制质心标记"""
        result = image.copy()
        
        for i in range(1, len(centers)):
            cv2.drawMarker(
                result,
                (int(centers[i][0]), int(centers[i][1])),
                self.config.marker_color,
                self.config.marker_type,
                self.config.marker_size,
                self.config.marker_thickness
            )
        
        return result
    
    def add_count_text(self, image: np.ndarray, count: int) -> np.ndarray:
        """添加计数文本"""
        result = image.copy()
        cv2.putText(
            result,
            f"count={count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.text_font_scale,
            self.config.text_color,
            self.config.text_thickness
        )
        return result
    
    def visualize_results(self, original_image: np.ndarray, 
                         analysis_results: Dict[str, Any]) -> np.ndarray:
        """创建最终的可视化结果"""
        self.logger.info("开始创建可视化结果")
        
        opencv_results = analysis_results['opencv']
        labels = opencv_results['labels']
        centers = opencv_results['centers']
        count = opencv_results['count']
        
        # 创建彩色连通域
        colored_output = self.create_colored_components(
            original_image, labels, opencv_results['num_labels']
        )
        
        # 图像权重叠加
        result = cv2.addWeighted(
            original_image, 
            self.config.result_alpha, 
            colored_output, 
            self.config.overlay_alpha, 
            0
        )
        
        # 绘制质心
        result = self.draw_centroids(result, centers)
        
        # 添加计数文本
        result = self.add_count_text(result, count)
        
        self.logger.info("可视化结果创建完成")
        return result


class CellImageProcessor:
    """细胞图像处理主控制器"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.preprocessor = ImagePreprocessor(self.config)
        self.segmentation = ImageSegmentation(self.config)
        self.analyzer = ConnectedComponentAnalyzer(self.config)
        self.visualizer = ResultVisualizer(self.config)
        
        # 设置日志
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def process_image(self, image_path: str, show_intermediate: bool = True, 
                     save_results: bool = True) -> Dict[str, Any]:
        """
        完整的图像处理流程
        
        Args:
            image_path: 输入图像路径
            show_intermediate: 是否显示中间结果
            save_results: 是否保存结果文件
            
        Returns:
            包含所有处理结果的字典
        """
        self.logger.info(f"开始处理图像: {image_path}")
        
        # 读取图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"无法读取图像文件: {image_path}")
        
        # 预处理
        preprocessed = self.preprocessor.preprocess(original_image)
        
        # 分割
        threshold_value, mask = self.segmentation.segment(preprocessed)
        
        # 连通域分析
        analysis_results = self.analyzer.analyze_components(mask)
        
        # 可视化
        final_result = self.visualizer.visualize_results(original_image, analysis_results)
        
        # 保存结果
        if save_results:
            self._save_results(threshold_value, mask, analysis_results, final_result)
        
        # 显示结果
        if show_intermediate:
            self._show_intermediate_results(original_image, preprocessed, mask, final_result)
        
        results = {
            'original_image': original_image,
            'preprocessed': preprocessed,
            'mask': mask,
            'analysis_results': analysis_results,
            'final_result': final_result,
            'threshold_value': threshold_value
        }
        
        self.logger.info("图像处理完成")
        return results
    
    def _save_results(self, threshold_value: float, mask: np.ndarray, 
                     analysis_results: Dict[str, Any], final_result: np.ndarray):
        """保存处理结果"""
        if not self.config.save_intermediate_results:
            return
        
        output_dir = self.config.output_dir
        
        # 保存二值化结果
        threshold_path = os.path.join(output_dir, self.config.threshold_filename)
        # 这里需要重新生成二值化图像用于保存
        _, binary_for_save = cv2.threshold(
            mask, 127, 255, cv2.THRESH_BINARY
        )
        cv2.imwrite(threshold_path, binary_for_save)
        
        # 保存mask
        mask_path = os.path.join(output_dir, self.config.mask_filename)
        np.savetxt(mask_path, mask)
        
        # 保存质心坐标
        centers_path = os.path.join(output_dir, self.config.centers_filename)
        centroids = analysis_results['skimage']['centroids']
        if centroids:
            np.savetxt(centers_path, centroids)
        
        self.logger.info("结果文件保存完成")
    
    def _show_intermediate_results(self, original: np.ndarray, preprocessed: np.ndarray,
                                 mask: np.ndarray, final_result: np.ndarray):
        """显示中间处理结果"""
        cv2.imshow("Original", original)
        cv2.imshow("Preprocessed", preprocessed)
        cv2.imshow("Mask", mask)
        cv2.imshow("Final Result", final_result)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# 向后兼容的函数接口
def show_img(ax_img, img, title, cmap="gray"):
    """保持原有的显示函数接口"""
    ax_img.imshow(img, cmap)
    ax_img.set_title(title)
    ax_img.set_axis_off()


def img2Gray(image):
    """保持原有的灰度转换函数接口"""
    preprocessor = ImagePreprocessor(ProcessingConfig())
    return preprocessor.to_grayscale(image, method="manual")


# 主程序入口（保持原有的执行方式）
if __name__ == "__main__":
    # 使用新的面向对象接口
    processor = CellImageProcessor()
    
    try:
        # 处理图像
        results = processor.process_image(
            r"G:\wxh\16ban\gcellcode\Cell-Code\Seg_enhance\iamges\img_new_pre2.jpg",
            show_intermediate=False,
            save_results=True
        )
        
        print(f"处理完成！发现 {results['analysis_results']['opencv']['count']} 个细胞")
        
    except FileNotFoundError as e:
        print(f"文件错误: {e}")
    except Exception as e:
        print(f"处理错误: {e}")