import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import Voronoi
from shapely.geometry import Polygon


def bounded_voronoi(bnd, pnts, r, output_image_path):

    gn_pnts = np.concatenate([pnts, np.array([[5000, 5000], [5000, -5000], [-60000, 0]])])

    vor = Voronoi(gn_pnts)


    bnd_poly = Polygon(bnd)
    vor_polys = []
    for i in range(len(gn_pnts) - 3):
        vor_poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
        i_cell = bnd_poly.intersection(Polygon(vor_poly))
        if i_cell.geom_type == 'Polygon':
            temp = list(i_cell.exterior.coords[:-1])
            temp = np.asarray(temp)
            if temp.shape[0] != 0:
                vor_polys.append(list(i_cell.exterior.coords[:-1]))
        elif i_cell.type == 'MultiPolygon':
            vor_polys.append(list(list(i_cell)[0].exterior.coords))
            vor_polys.append(list(list(i_cell)[1].exterior.coords))

    fig1 = plt.figure(figsize=(7, 6))
    ax = fig1.add_subplot(111)

    ax.scatter(pnts[:, 0], pnts[:, 1], c=(0/255, 255/255, 255/255))
    poly_vor = PolyCollection(vor_polys, edgecolor="red",
                              facecolors="None", linewidth=r)

    ax.add_collection(poly_vor)

    xmin = np.min(bnd[:, 0])
    xmax = np.max(bnd[:, 0])
    ymin = np.min(bnd[:, 1])
    ymax = np.max(bnd[:, 1])

    ax.set_xlim(xmin - 10, xmax + 10)
    ax.set_ylim(ymin - 10, ymax + 10)
    ax.set_aspect('equal')

    plt.gca().invert_yaxis()
    plt.savefig(output_image_path)

    return vor_polys

def generate_voronoi_diagram(threshold_image_path, centers_file_path, output_json_path,output_image_path):

    src = cv2.imread(threshold_image_path)
    h, w = src.shape[:2]
    print(src.shape)

    r = 2
    bnd = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    pnts = np.loadtxt(centers_file_path)
    print('pnts', pnts.shape)
    p_d = []
    for i in range(0, len(pnts) - 1):
        point1 = pnts[i]
        point2 = pnts[i + 1]
        precise = 1000
        op = np.linalg.norm(point1 - point2)
        if op < 30:
            line_points = np.linspace(point1, point2, precise)
            d = 0
            for j in range(len(line_points)):
                if np.any(src[int(line_points[j][1])][int(line_points[j][0])] == 255):
                    d = d + 1
            temp = d * op / precise
            p_d.append(temp)
    dis = np.mean(p_d)
    print('p_d', dis)

    vor_polys = bounded_voronoi(bnd, pnts, r,output_image_path)

    print("vor_polys", len(vor_polys))
    print(vor_polys[0])

    json_file_path = output_json_path
    json_file = open(json_file_path, mode='w')

    save_json_content = []
    for img_name in vor_polys:
        t = vor_polys.index(img_name)
        point = pnts[t].tolist()
        result_json = {
            "region id": t,
            "vertices": img_name,
            "point id": t,
            "point": point,
        }
        save_json_content.append(result_json)

    json.dump(save_json_content, json_file, indent=4)

if __name__ == '__main__':
    threshold_image_path = r'C:\wxh\mycode\gcellcode\result\N00007789\thres1.jpg'
    centers_file_path = r'C:\wxh\mycode\gcellcode\result\N00007789\centers.txt'
    output_json_path = r'C:\wxh\mycode\gcellcode\result\N00007789\N00007789.json'
    output_image_path = r'C:\wxh\mycode\gcellcode\result\N00007789\output_figure.png'
    generate_voronoi_diagram(threshold_image_path, centers_file_path, output_json_path,output_image_path)