import os.path

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
import cv2
from tqdm import tqdm

from Detection import detection
from Tamplate import tamplate_data
from Segment import character_segmentation

args = {'plate_output': 'plates',
        'segmentation_output': 'segmentation',
        'binary_threshold': 95,
        'char_segment_threshold': 195,
        'edge_algorithm': 'sobel',
        'region_threshold': 1000,
        'char_region_threshold': (500, 5000),  # 字符区域
        'aspect_rate_threshold': (0.5, 4.5),  # 字符高宽比
        'kernel_size': (101, 81),
        'stride': (4, 4)}

image_path = 'Text recognition.jpg'


def extract(image_path, args=None):
    image = np.array(Image.open(image_path).convert('L'))  # 转换为灰度图
    # edge_picture = Image.fromarray(image)
    # edge_picture.save('gray_image.png')
    if not os.path.exists(args['plate_output']):
        os.mkdir(args['plate_output'])

    if not os.path.exists(args['segmentation_output']):
        os.mkdir(args['segmentation_output'])

    # --------边缘检测-----------

    if args['edge_algorithm'] == 'canny':
        edge = feature.canny(image, sigma=0.5)
    if args['edge_algorithm'] == 'sobel':
        x = cv2.Sobel(image, cv2.CV_16S, 1, 0)  # x方向一阶导
        y = cv2.Sobel(image, cv2.CV_16S, 0, 1)  # x方向一阶导

        absX = cv2.convertScaleAbs(x)  # 转回uint8  由于会出现负值的情况，因此使用cv2.convertScalerAbs() 转换为绝对值的形式
        absY = cv2.convertScaleAbs(y)

        edge = cv2.addWeighted(absX, 0.5, absY, 0.5, 0) > 50  # 图像融合
    # edge_picture = Image.fromarray(edge)
    # edge_picture.save('edge.png')

    # --------------找出车牌区域----------

    patches = detection(image, edge, args)

    # -------------模板匹配------------
    feature_dict = tamplate_data(args=args)  # 字符的特征
    args['feature_dict'] = feature_dict

    for i in range(len(patches)):
        patch = -(patches[i].astype(np.float)) + 255
        patch = 255 * (patch - np.min(patch)) / (np.max(patch) - np.min(patch))
        character_segmentation(patch, args, id=i)

extract(image_path, args)
