from PIL import Image

def compare_and_save_images(image1_path, image2_path, output_path):
    # 打开两张图片
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # 确保两张图片大小一致
    if image1.size != image2.size:
        raise ValueError("两张图片的大小不一致")

    # 转换为灰度图像
    gray_image1 = image1.convert('L')
    gray_image2 = image2.convert('L')

    # 获取像素数据
    pixels1 = gray_image1.load()
    pixels2 = gray_image2.load()

    # 创建新的Image对象用于保存结果
    result_image = Image.new('L', image1.size)
    result_pixels = result_image.load()

    # 比较每个像素位置的灰度值，取灰度值最大的值
    for x in range(image1.width):
        for y in range(image1.height):
            result_pixels[x, y] = max(pixels1[x, y], pixels2[x, y])

    # 保存结果图片
    result_image.save(output_path)

if __name__ == "__main__":
    # 替换以下路径为你的图片路径
    image1_path = "Test_other_images/N00007788/N00007788_448x304_pre.jpg"
    image2_path = "Test_other_images/N00007788/N00007788_X_X_fli_pre.jpg"
    output_path = "Test_other_images/N00007788/N00007788_new_pre.jpg"

    compare_and_save_images(image1_path, image2_path, output_path)
