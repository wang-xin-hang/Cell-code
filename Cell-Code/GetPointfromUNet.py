import numpy as np
import cv2
from skimage import measure


# 显示图像
def show_img(ax_img, img, title, cmap="gray"):
    ax_img.imshow(img, cmap)
    ax_img.set_title(title)
    ax_img.set_axis_off()

# 将彩色图像转为色调图像
def img2Gray(image):
    # 加权平均分
    h, w = image.shape[:2]
    gray5 = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            gray5[i, j] = 0.2125 * image[i, j][2] + 0.7154 * image[i, j][0] + 0.0721 * image[i, j][1]
    return gray5

img = cv2.imread(r"Seg_enhance/Test_other_images/N00007788/N00007788_new_pre.jpg")
print(img)
#灰度图二值化
img_gray = img2Gray(img)
cv2.imshow("img_gray", img_gray)
# 高斯差分
img_Gauss = cv2.GaussianBlur(img_gray, (5, 5), 5)
cv2.imshow("img_Gauss", img_Gauss)
# 均衡（增强对比度）
# img_equilibrium = cv2.equalizeHist(img_Gauss)
# cv2.imshow("img_equilibrium", img_equilibrium)

# ret, thres = cv2.threshold(img_Gauss, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

ret, thres = cv2.threshold(img_Gauss, 40, 255, cv2.THRESH_BINARY)
cv2.imshow('thres', thres)
cv2.imwrite("thres1.jpg", thres)

# 膨胀操作
kernel_dilate = np.ones((3, 3), np.uint8)
dilation = cv2.dilate(thres, kernel_dilate, iterations = 1)
cv2.imshow('dilation', dilation)

#腐蚀操作
# kernel_erode = np.ones((3,3),np.uint8)
# erosion = cv2.erode(dilation, kernel_erode, iterations = 1)
# cv2.imshow('erosion', erosion)

# 取反
mask_inv = cv2.bitwise_not(dilation)
mask_inv[:, 0:5] = 0  # 左边界
mask_inv[:, -1:-6:-1] = 0# 右边界
mask_inv[0:5, :] = 0  # 上边界
mask_inv[-1:-6:-1, :] = 0  # 下边界

# print(mask_inv)
np.savetxt("mask_inv.txt", mask_inv)
cv2.imshow('mask_inv', mask_inv)

labels = measure.label(mask_inv, connectivity=2)
# 将包含每个连通区域的信息，如面积、质心坐标
properties = measure.regionprops(labels)
valid_label = set()
# # 区域内像素点总数
area = []
# # 质心坐标
centroid = []
for i in range(np.max(labels)):  # 连通区域个数
    area.append(properties[i].area)
    centroid.append([properties[i].centroid[1], properties[i].centroid[0]])

# print(centroid)
# print('centers =', len(centroid))
np.savetxt("centers.txt", centroid)

# # 连通域分析
num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(mask_inv, connectivity=8)

# 不同的连通域赋予不同的颜色
output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
for i in range(1, num_labels):
    mask = labels == i
    output[:, :, 0][mask] = np.random.randint(0, 255)
    output[:, :, 1][mask] = np.random.randint(0, 255)
    output[:, :, 2][mask] = np.random.randint(0, 255)

result = cv2.addWeighted(img, 0.8, output, 0.5, 0)  # 图像权重叠加
for i in range(1, len(centers)):
    cv2.drawMarker(result, (int(centers[i][0]), int(centers[i][1])), (0, 0, 255), 1, 2, 2)
#
cv2.putText(result, "count=%d" % (len(centers) - 1), (20, 30), 0, 0.75, (0, 255, 0), 2)
cv2.imshow('result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
