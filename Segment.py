from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
import cv2
import ipdb
from tqdm import tqdm
from skimage import morphology, draw

from Tamplate import feature_extract


def character_segmentation(plate, args, id=1):
    patches = []
    pred_char = ''
    patch_id = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

    thresh, image = cv2.threshold(plate, args['char_segment_threshold'], 255, cv2.THRESH_BINARY)

    image = cv2.erode(image, kernel)  # 腐蚀
    image = cv2.dilate(image, kernel)  # 膨胀

    image = cv2.dilate(image, kernel)  # 膨胀
    image = cv2.erode(image, kernel)  # 腐蚀
    # edge_picture = Image.fromarray(image)
    # edge_picture.show()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image.astype(np.uint8), 4)  # 连通域
    sort_index = np.argsort(stats[:, 0])

    for i in range(num_labels):
        patch_id += 1

        width = stats[sort_index[i]][2]
        height = stats[sort_index[i]][3]
        aspect_rate = height / width
        region_area = height * width
        # print('id:{}, patch id:{}   region_area:{}  aspect_rate:{}'.format(id + 1, patch_id, region_area, aspect_rate))
        if (region_area >= args['char_region_threshold'][0]) \
                and (region_area <= args['char_region_threshold'][1]) \
                and (aspect_rate > args['aspect_rate_threshold'][0]) \
                and (aspect_rate < args['aspect_rate_threshold'][1]):
            x0 = stats[sort_index[i]][0]
            y0 = stats[sort_index[i]][1]

            expand_size = 0
            patch = image[max(y0 - expand_size, 0): min(y0 + height + expand_size, image.shape[0] - 1),
                    max(x0 - expand_size, 0): min(x0 + width + expand_size, image.shape[1] - 1)].copy()

            patches.append(patch)
            Image.fromarray(patch.astype(np.uint8)).save(os.path.join(
                'segment_char', 'id_{}_patch_{}.jpg'.format(id + 1, patch_id + 1)))

            patch_feature, patch_sket = feature_extract(patch, vision=False)  # 提取
            # rotation = compute_rotation()

            pred, similarity = patch_recognation(patch, patch_feature, args)
            pred_char += pred

            Image.fromarray(patch_sket.astype(np.uint8)).save(os.path.join(
                args['segmentation_output'], 'id_{}_patch_{}_{}.jpg'.format(id + 1, patch_id + 1, pred)))

    # plt.figure(0)
    # plt.imshow(plate, cmap='gray')
    # plt.title(pred_char)
    # plt.savefig(os.path.join(args['segmentation_output'], '{}.jpg'.format(id + 1)))


def patch_recognation(patch, patch_feature, args):
    feature_dict = args['feature_dict']

    pred = None
    best_similarity = 255
    for char_gt, feature_gt in feature_dict.items():
        for temp_feature in feature_gt:
            similarity = np.mean((np.abs(patch_feature['pooling'] - temp_feature['pooling'])))  # 计算字符模板的特征与车牌字符特称的距离
            # print(char_gt, similarity)
            if best_similarity > similarity:
                best_similarity = similarity
                pred = char_gt

    return pred, best_similarity
