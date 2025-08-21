from PIL import Image
import numpy as np
import logging
from typing import Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class ImageEnhanceConfig:
    """图像增强配置类，管理所有处理参数"""
    
    # 输出格式配置
    output_format: str = "JPEG"
    output_quality: int = 95
    
    # 处理选项
    use_vectorized: bool = True  # 是否使用向量化操作
    validate_inputs: bool = True  # 是否验证输入
    create_backup: bool = False  # 是否创建备份
    
    # 日志配置
    enable_logging: bool = True
    log_level: str = "INFO"


class ImageValidator:
    """图像验证器，负责输入验证和错误检查"""
    
    def __init__(self, config: ImageEnhanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_file_exists(self, file_path: Union[str, Path]) -> Path:
        """验证文件是否存在"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"图像文件不存在: {path}")
        if not path.is_file():
            raise ValueError(f"路径不是文件: {path}")
        return path
    
    def validate_image_format(self, image_path: Union[str, Path]) -> bool:
        """验证图像格式是否支持"""
        try:
            with Image.open(image_path) as img:
                # 尝试验证图像
                img.verify()
            return True
        except Exception as e:
            raise ValueError(f"无效的图像文件 {image_path}: {e}")
    
    def validate_images_compatible(self, image1: Image.Image, image2: Image.Image) -> bool:
        """验证两张图像是否兼容（大小一致）"""
        if image1.size != image2.size:
            raise ValueError(
                f"两张图片的大小不一致: {image1.size} vs {image2.size}"
            )
        return True
    
    def validate_output_path(self, output_path: Union[str, Path]) -> Path:
        """验证输出路径"""
        path = Path(output_path)
        
        # 确保输出目录存在
        path.parent.mkdir(parents=True, exist_ok=True)
        
        return path


class ImageProcessor:
    """图像处理器，负责核心的图像处理逻辑"""
    
    def __init__(self, config: ImageEnhanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """加载图像"""
        try:
            image = Image.open(image_path)
            self.logger.info(f"成功加载图像: {image_path}, 大小: {image.size}")
            return image
        except Exception as e:
            raise ValueError(f"无法加载图像 {image_path}: {e}")
    
    def convert_to_grayscale(self, image: Image.Image) -> Image.Image:
        """转换为灰度图像"""
        if image.mode == 'L':
            return image
        
        gray_image = image.convert('L')
        self.logger.debug(f"图像转换为灰度，模式: {image.mode} -> L")
        return gray_image
    
    def compare_pixels_vectorized(self, image1: Image.Image, image2: Image.Image) -> Image.Image:
        """使用向量化操作比较像素（性能优化版本）"""
        # 转换为numpy数组进行向量化操作
        array1 = np.array(image1)
        array2 = np.array(image2)
        
        # 取每个位置的最大值
        result_array = np.maximum(array1, array2)
        
        # 转换回PIL图像
        result_image = Image.fromarray(result_array, mode='L')
        
        self.logger.info("使用向量化操作完成像素比较")
        return result_image
    
    def compare_pixels_traditional(self, image1: Image.Image, image2: Image.Image) -> Image.Image:
        """使用传统循环比较像素（保持原有逻辑）"""
        # 获取像素数据
        pixels1 = image1.load()
        pixels2 = image2.load()
        
        # 创建新的Image对象用于保存结果
        result_image = Image.new('L', image1.size)
        result_pixels = result_image.load()
        
        # 比较每个像素位置的灰度值，取灰度值最大的值
        for x in range(image1.width):
            for y in range(image1.height):
                result_pixels[x, y] = max(pixels1[x, y], pixels2[x, y])
        
        self.logger.info("使用传统循环完成像素比较")
        return result_image
    
    def compare_images(self, image1: Image.Image, image2: Image.Image) -> Image.Image:
        """比较两张图像，返回增强后的结果"""
        # 转换为灰度图像
        gray_image1 = self.convert_to_grayscale(image1)
        gray_image2 = self.convert_to_grayscale(image2)
        
        # 选择处理方法
        if self.config.use_vectorized:
            result = self.compare_pixels_vectorized(gray_image1, gray_image2)
        else:
            result = self.compare_pixels_traditional(gray_image1, gray_image2)
        
        return result
    
    def save_image(self, image: Image.Image, output_path: Union[str, Path]) -> None:
        """保存图像"""
        try:
            save_kwargs = {}
            
            # 根据格式设置保存参数
            if self.config.output_format.upper() == 'JPEG':
                save_kwargs['quality'] = self.config.output_quality
                save_kwargs['optimize'] = True
            
            image.save(output_path, format=self.config.output_format, **save_kwargs)
            self.logger.info(f"图像已保存到: {output_path}")
            
        except Exception as e:
            raise ValueError(f"保存图像失败 {output_path}: {e}")


class ImageEnhancer:
    """图像增强器主控制类"""
    
    def __init__(self, config: Optional[ImageEnhanceConfig] = None):
        self.config = config or ImageEnhanceConfig()
        self.validator = ImageValidator(self.config)
        self.processor = ImageProcessor(self.config)
        
        # 设置日志
        if self.config.enable_logging:
            self._setup_logging()
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_logging(self):
        """设置日志配置"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def enhance_images(self, image1_path: Union[str, Path], 
                      image2_path: Union[str, Path], 
                      output_path: Union[str, Path]) -> None:
        """
        增强两张图像并保存结果
        
        Args:
            image1_path: 第一张图像路径
            image2_path: 第二张图像路径
            output_path: 输出图像路径
        """
        self.logger.info("开始图像增强处理")
        
        try:
            # 验证输入
            if self.config.validate_inputs:
                self.validator.validate_file_exists(image1_path)
                self.validator.validate_file_exists(image2_path)
                self.validator.validate_image_format(image1_path)
                self.validator.validate_image_format(image2_path)
                output_path = self.validator.validate_output_path(output_path)
            
            # 加载图像
            image1 = self.processor.load_image(image1_path)
            image2 = self.processor.load_image(image2_path)
            
            # 验证图像兼容性
            if self.config.validate_inputs:
                self.validator.validate_images_compatible(image1, image2)
            
            # 创建备份
            if self.config.create_backup and Path(output_path).exists():
                backup_path = Path(output_path).with_suffix('.backup' + Path(output_path).suffix)
                Path(output_path).rename(backup_path)
                self.logger.info(f"已创建备份: {backup_path}")
            
            # 处理图像
            result_image = self.processor.compare_images(image1, image2)
            
            # 保存结果
            self.processor.save_image(result_image, output_path)
            
            self.logger.info("图像增强处理完成")
            
        except Exception as e:
            self.logger.error(f"图像增强处理失败: {e}")
            raise
        
        finally:
            # 清理资源
            try:
                if 'image1' in locals():
                    image1.close()
                if 'image2' in locals():
                    image2.close()
                if 'result_image' in locals():
                    result_image.close()
            except:
                pass
    
    def batch_enhance(self, image_pairs: list, output_dir: Union[str, Path]) -> None:
        """
        批量处理多对图像
        
        Args:
            image_pairs: 图像对列表，每个元素为(image1_path, image2_path, output_name)
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"开始批量处理 {len(image_pairs)} 对图像")
        
        success_count = 0
        for i, (img1_path, img2_path, output_name) in enumerate(image_pairs, 1):
            try:
                output_path = output_dir / output_name
                self.logger.info(f"处理第 {i}/{len(image_pairs)} 对图像")
                
                self.enhance_images(img1_path, img2_path, output_path)
                success_count += 1
                
            except Exception as e:
                self.logger.error(f"处理第 {i} 对图像失败: {e}")
        
        self.logger.info(f"批量处理完成: {success_count}/{len(image_pairs)} 成功")


# 向后兼容的函数接口
def compare_and_save_images(image1_path: Union[str, Path], 
                           image2_path: Union[str, Path], 
                           output_path: Union[str, Path]) -> None:
    """
    Args:
        image1_path: 第一张图像路径
        image2_path: 第二张图像路径
        output_path: 输出图像路径
    """
    # 使用默认配置创建增强器
    config = ImageEnhanceConfig(enable_logging=False)  # 保持原有的静默模式
    enhancer = ImageEnhancer(config)
    
    # 调用新的增强方法
    enhancer.enhance_images(image1_path, image2_path, output_path)



# 主程序入口（保持原有的执行方式）
if __name__ == "__main__":
    # 使用原有的路径配置
    image1_path = r"G:\wxh\16ban\gcellcode\Cell-Code\img_new_pre.jpg"
    image2_path = r"G:\wxh\16ban\gcellcode\Cell-Code\img_fli.jpg"
    output_path = r"G:\wxh\16ban\gcellcode\Cell-Code\img_new_pre2.jpg"
    
    try:
        # 方式1: 使用原有的函数接口（向后兼容）
        print("使用原有接口处理图像...")
        compare_and_save_images(image1_path, image2_path, output_path)
        print("处理完成！")

    except FileNotFoundError as e:
        print(f"文件错误: {e}")
        print("请确保图像文件存在于指定路径")
    except Exception as e:
        print(f"处理错误: {e}")