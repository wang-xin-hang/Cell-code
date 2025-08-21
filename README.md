# Cell-Code

细胞图像分析与处理工具集，包含细胞检测、维诺图生成和深度学习分割模型。

## 项目简介

Cell-Code是一个用于细胞图像分析的工具集，主要功能包括：

- 细胞中心点检测
- 维诺图生成与可视化
- 图像增强处理
- 基于深度学习的细胞分割

该项目分为两个主要部分：图像处理模块和深度学习模块。

## 功能模块

### 图像处理模块 (Cell-Code/)

- **GetPointfromUNet.py**: 从分割图像中提取细胞中心点
- **Voronoi Graph.py**: 基于细胞中心点生成有界维诺图
- **Pixel_cropping.py**: 裁剪维诺图图像
- **sobel.py**: 图像增强和掩膜叠加
- **rotation_image.py**: 图像旋转处理
- **Flipud_img.py**: 图像翻转处理

### 深度学习模块 (Cell-Code-Deep-learning/)

- **denseunet_test_main.py**: DenseUNet模型测试主程序
- 预训练权重文件：包含多种DenseUNet变体的权重

## 安装说明

### 环境要求

- Python 3.8+
- 依赖包列表见requirements.txt

### 安装步骤

1. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### 细胞检测与维诺图生成流程

运行 main.py

## 许可证

本项目采用MIT许可证，详见[LICENSE](LICENSE)文件。
