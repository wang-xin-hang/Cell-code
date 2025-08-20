import json
from math import sqrt

import cv2
from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon


# newV:返回两点间的方向向量（未单位化）
def getPositionVector(x1, x2):
    newV = [x1[0] - x2[0], x1[1] - x2[1]]
    return newV


# 返回单位化后的向量
def getUnitVector(x):
    size = sqrt(pow(x[0], 2) + pow(x[1], 2))
    # print("向量长度", size)
    if size != 0:
        for i in range(2):
            x[i] = x[i] / size
    return x


# 根据单位向量,点,长度获取下一个点
def getNextPoint(raw_point, dir_Vector, v):
    newV = [raw_point[0] + dir_Vector[0] * v, raw_point[1] + dir_Vector[1] * v]
    return newV


# 根据两点得到向量
def getVector(a2, a1):
    p_Vector = getPositionVector(a2, a1)
    p_Vector = getUnitVector(p_Vector)
    return p_Vector


def bounded_voronoi(bnd, pnts, r):
    """
    有界なボロノイ図を計算?描画する関数．
    """

    # すべての母点のボロノイ領域を有界にするために，ダミー母点を3個追加
    gn_pnts = np.concatenate([pnts, np.array([[5000, 5000], [5000, -5000], [-60000, 0]])])
    # gn_pnts = np.concatenate([pnts, np.array([[100, 100], [100, -100], [-100, 0]])])

    # ボロノイ図の計算
    vor = Voronoi(gn_pnts)
    vor1 = Voronoi(pnts, furthest_site=False, incremental=True, qhull_options=None)
    # fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='black',
    #                       line_width=2, line_alpha=0.6, point_size=1)
    # fig = voronoi_plot_2d(vor1, show_vertices=False, line_colors='black',
    #                       line_width=5, line_alpha=0.6, point_size=1)
    # fig1 = plt.figure(figsize=(7, 6))
    # ax1 = fig.add_subplot(111)
    # plt.show()

    # 定义大小
    bnd_poly = Polygon(bnd)
    # print("bnd_poly", bnd_poly)
    vor_polys = []
    for i in range(len(gn_pnts) - 3):
        # 不考虑封闭空间的波洛诺伊区域
        vor_poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
        # print('vor_polys', vor_polys)
        # 将分割的区域分解，计算区域的公共部分
        # i_cell = bnd_poly.intersection(Polygon(vor_poly).buffer(0.001))
        # 计算出所有点
        i_cell = bnd_poly.intersection(Polygon(vor_poly))
        # print('i_cell', i_cell)
        # vor_polys.append(list(i_cell.exterior.coords[:-1]))
        # # 顶点坐标
        if i_cell.geom_type == 'Polygon':
            temp = list(i_cell.exterior.coords[:-1])
            temp = np.asarray(temp)
            if temp.shape[0] != 0:
                vor_polys.append(list(i_cell.exterior.coords[:-1]))
        elif i_cell.type == 'MultiPolygon':
            vor_polys.append(list(list(i_cell)[0].exterior.coords))
            vor_polys.append(list(list(i_cell)[1].exterior.coords))

            # 定义画布
    fig1 = plt.figure(figsize=(7, 6))
    ax = fig1.add_subplot(111)

    # 中心点
    ax.scatter(pnts[:, 0], pnts[:, 1], c=(0/255, 255/255, 255/255))
    poly_vor = PolyCollection(vor_polys, edgecolor="red",
                              facecolors="None", linewidth=r)
    # print(pnts[:, 0], pnts[:, 1])

    ax.add_collection(poly_vor)

    xmin = np.min(bnd[:, 0])
    xmax = np.max(bnd[:, 0])
    ymin = np.min(bnd[:, 1])
    ymax = np.max(bnd[:, 1])

    # ax.set_xlim(xmin - 0.1, xmax + 0.1)
    # ax.set_ylim(ymin - 0.1, ymax + 0.1)
    ax.set_xlim(xmin - 10, xmax + 10)
    ax.set_ylim(ymin - 10, ymax + 10)
    ax.set_aspect('equal')
    # plt.text(122.98495445037038, 292.1393398613072, "*", color='y')
    # plt.text(154.8697596042729, 277.3776988133825, "*", color='y')
    # plt.text(156.3196078207491, 233.91049143738394, "*", color='y')
    # plt.text(110.23277732462263, 237.47442953070077, "*", color='y')
    # plt.text(96.51359859249406, 261.24700511420536, "*", color='y')
    #
    # plt.text(40.0, 419.0, "2", color='r')
    # plt.text(132.0, 261.0, ".", color='r')
    # plt.text(86.51384083044982, 261.3166089965398, "*", color='b')
    # for i in range(len(linents)):
    #     plt.text(linents[i][0], linents[i][1], ".", color='b')

    # print(ax)
    plt.gca().invert_yaxis()
    plt.savefig('output_figure.png')
    plt.show()

    return vor_polys


src = cv2.imread(r"thres1.jpg")
# src = cv2.imread(r"E:\111\datas\jpg\0524\N00001858-SDC2.png")
h, w = src.shape[:2]
print(src.shape)

r = 2
# 446, 304
# bnd = np.array([[125, 50], [250, 50], [250, 250], [125, 250]])
bnd = np.array([[0, 0], [w, 0], [w, h], [0, h]])
# bnd = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
# 把点归一化 298 152
# n = 5
# pnts = np.random.rand(n, 2)
# pnts = np.loadtxt(r"E:\111\datas\jpg\PKP4.txt")
pnts = np.loadtxt("centers.txt")
# pnts = np.loadtxt("l.txt")
print('pnts', pnts.shape)
p_d = []
for i in range(0, len(pnts) - 1):
    point1 = pnts[i]
    point2 = pnts[i + 1]
    # vector1 = np.asarray(pnts[i][0])
    # vector2 = np.asarray(pnts[i][0])
    precise = 1000
    op = np.linalg.norm(point1 - point2)
    # print(point1,point2,'两点之间的距离:', op)
    if op < 30:
        line_points = np.linspace(point1, point2, precise)
        d = 0
        for j in range(len(line_points)):
            # print(int(line_points[j][1]))
            if np.any(src[int(line_points[j][1])][int(line_points[j][0])] == 255):
                d = d + 1
        temp = d * op / precise
        p_d.append(temp)
        # print('d=', temp)
dis = np.mean(p_d)
print('p_d', dis)

vor_polys = bounded_voronoi(bnd, pnts, r)

print("vor_polys", len(vor_polys))
print(vor_polys[0])

json_file_path = r'./Seg_enhance/Test_other_images/N00007788/N00007788.json'
json_file = open(json_file_path, mode='w')

save_json_content = []
for img_name in vor_polys:
    t = vor_polys.index(img_name)
    point = pnts[t].tolist()
    # new_points = []
    # for i in range(len(img_name)):
    #     point1 = img_name[i]
    #     vector = getVector(point, point1)
    #     temp = getNextPoint(point1, vector, dis/2)
    #     new_points.append(temp)
    # print(new_points)
    result_json = {
        "region id": t,
        "vertices": img_name,
        "point id": t,
        "point": point,
    }
    save_json_content.append(result_json)

json.dump(save_json_content, json_file, indent=4)
# np.savetxt("vor_polys.txt",np.asarray(vor_polys))

# points = np.array([[0, 0], [0, 1], [0, 2],
#                    [1, 0], [1, 1], [1, 2],
#                    [2, 0], [2, 1], [2, 2]])
# vor = Voronoi(points)
