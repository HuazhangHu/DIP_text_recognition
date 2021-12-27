from PIL import Image
import numpy as np
import os
import cv2


def detection(original_image, edge, args):
    patches = []
    id = 0
    kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 十字核函数
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))   # 矩形核函数

    # edge = cv2.erode(edge.astype(np.float), kernel3)
    # edge = cv2.dilate(edge.astype(np.float), kernel5)
    _, image_binary = cv2.threshold(original_image, args['binary_threshold'], 255, cv2.THRESH_BINARY)  # 图像二值化
    # edge_picture = Image.fromarray(image_binary)
    # edge_picture.save('binary_image.png')
    image = cv2.erode(image_binary, kernel3)  #腐蚀
    image = cv2.dilate(image, kernel3)  # 膨胀


    image = image - edge * image_binary
    picture = Image.fromarray(image)
    picture.save('modify.png')
    # 一幅图像进行连通域提取，并返回找到的连通域的信息,找出联通区域，4联通，stats表示x、y、width、height和面积
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image.astype(np.uint8), 4)

    sort_index = np.argsort(- stats[:, 4])
    # 按连通域从大到小排序
    for i in range(1, 50):
        width = stats[sort_index[i]][2]
        height = stats[sort_index[i]][3]
        if stats[sort_index[i]][4] >= args['region_threshold']:
            id += 1
            x0 = stats[sort_index[i]][0]
            y0 = stats[sort_index[i]][1]  # 左上角的坐标
            patch = original_image[y0: y0 + height, x0: x0 + width].copy()
            patches.append(patch)

            Image.fromarray(patch.astype(np.uint8)).\
                save(os.path.join(args['plate_output'], '{}_patch.jpg'.format(id)))
            # edge_picture = Image.fromarray(patch)
            # edge_picture.show()


    return patches

